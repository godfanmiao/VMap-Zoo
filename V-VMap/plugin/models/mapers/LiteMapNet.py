import torch
import math
from collections import OrderedDict
from mmcv.runner import force_fp32
from mmdet3d.models.builder import (build_backbone, build_head,build_neck)
from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from .base_mapper import BaseMapper, MAPPERS
from copy import deepcopy
from mmdet.models import build_loss
import numpy as np

@MAPPERS.register_module()
class LiteMapNet(BaseMapper):

    def __init__(self,
                 num_classes = 3,
                 num_queries=100,
                 backbone_cfg=dict(),
                 neck_cfg=dict(),
                 ipm_cfg=dict(),
                 head_cfg=dict(),
                 loss_cls=dict(),
                 loss_reg=dict(),
                 assigner=dict(),
                 pretrained=None,
                 sync_cls_avg_factor=True,
                 **kwargs):
        super().__init__()
  
        self.backbone = build_backbone(backbone_cfg)

        self.neck = build_neck(neck_cfg)
        self.ipm = build_neck(ipm_cfg)
        self.num_classes = num_classes
        self.head = build_head(head_cfg)
        self.num_queries = num_queries
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.assigner = build_assigner(assigner)
        
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        
        self.sync_cls_avg_factor = sync_cls_avg_factor
        
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            checkpoint_model = checkpoint['model']
            checkpoint_model = self.remap_checkpoint_keys(checkpoint_model)
            # load pre-trained model
            msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
            print(msg)
        else:
            pass

    def remap_checkpoint_keys(self,ckpt):
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('encoder'):
                k = '.'.join(k.split('.')[1:])  # remove encoder in the name
            if k.endswith('kernel'):
                k = '.'.join(k.split('.')[:-1])  # remove kernel in the name
                new_k = k + '.weight'
                if len(v.shape) == 3:  # resahpe standard convolution
                    kv, in_dim, out_dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[new_k] = v.permute(2, 1, 0). \
                        reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                elif len(v.shape) == 2:  # reshape depthwise convolution
                    kv, dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[new_k] = v.permute(1, 0). \
                        reshape(dim, 1, ks, ks).transpose(3, 2)
                continue
            elif 'ln' in k or 'linear' in k:
                k = k.split('.')
                k.pop(-2)  # remove ln and linear in the name
                new_k = '.'.join(k)
            else:
                new_k = k
            new_ckpt[new_k] = v

        # reshape grn affine parameters and biases
        for k, v in new_ckpt.items():
            if k.endswith('bias') and len(v.shape) != 1:
                new_ckpt[k] = v.reshape(-1)
            elif 'grn' in k:
                new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
        return new_ckpt

    def extract_img_feat(self, imgs):
        '''
            Extract image feaftures and sum up into one pic
            Args:
                imgs: B, n_cam, C, iH, iW
            Returns:
                img_feat: B * n_cam, C, H, W
        '''

        B, n_cam, C, iH, iW = imgs.shape
        imgs = imgs.view(B * n_cam, C, iH, iW)

        img_feats = self.backbone(imgs)

        # reduce the channel dim
        img_feat = self.neck(img_feats)

        return img_feat

    def forward_train(self, img, vectors, points=None, img_metas=None, **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]
        Out:
            loss, log_vars, num_sample
        '''
        #  prepare labels and images

        gts, img, valid_idx, points = self.batch_data(
            vectors, img, img.device, points)

        img_feat = self.extract_img_feat(img)
        
        ego2cam = []
        for img_meta in img_metas:
            ego2cam.append(img_meta['ego2img'])
        ego2cam = np.asarray(ego2cam)
        
        img_shape = img.shape[-2:]
        # Neck
        bev_feats = self.ipm(img_feat, ego2cam, img_shape)

        outputs_classes, outputs_coords = self.head(bev_feats)
        outputs = []
        for outputs_class, outputs_coord in zip(outputs_classes, outputs_coords):
            reg_points_list = []
            scores_list = []
            for scores, reg_point in zip(outputs_class, outputs_coord):
                reg_points_list.append(reg_point)
                scores_list.append(scores)

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list
            }
            outputs.append(pred_dict)

        loss_dict, det_match_idxs, det_match_gt_idxs, gt_lines_list = self.loss(gts=gts, preds=outputs)

        # format loss
        loss = 0
        for name, var in loss_dict.items():
            loss = loss + var

        # update the log
        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({'total': loss.item()})

        num_sample = img.size(0)

        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, **kwargs):
        '''
            inference pipeline
        '''

        #  prepare labels and images
        
        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        img_feat = self.extract_img_feat(img)

        ego2cam = []
        for img_meta in img_metas:
            ego2cam.append(img_meta['ego2img'])
        ego2cam = np.asarray(ego2cam)

        img_shape = img.shape[-2:]
        # Neck
        bev_feats = self.ipm(img_feat, ego2cam, img_shape)

        outputs_classes, outputs_coords = self.head(bev_feats)
        outputs = []
        for outputs_class, outputs_coord in zip(outputs_classes, outputs_coords):
            reg_points_list = []
            scores_list = []
            prop_mask_list = []
            for scores, reg_point in zip(outputs_class, outputs_coord):
                reg_points_list.append(reg_point)
                scores_list.append(scores)
                prop_mask = outputs_class.new_ones((len(scores),), dtype=torch.bool)
                prop_mask[-self.num_queries:] = False
                prop_mask_list.append(prop_mask)

            outputs = []
            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list,
                'prop_mask': prop_mask_list
            }
            outputs.append(pred_dict)

        # take predictions from the last layer
        preds_dict = outputs[-1]

        results_list = self.post_process(preds_dict, tokens)

        return results_list

    def batch_data(self, vectors, imgs, device, points=None):
        bs = len(vectors)
        # filter none vector's case
        num_gts = []
        for idx in range(bs):
            num_gts.append(sum([len(v) for k, v in vectors[idx].items()]))
        valid_idx = [i for i in range(bs) if num_gts[i] > 0]
        assert len(valid_idx) == bs # make sure every sample has gts

        all_labels_list = []
        all_lines_list = []
        for idx in range(bs):
            labels = []
            lines = []
            for label, _lines in vectors[idx].items():
                for _line in _lines:
                    labels.append(label)
                    if len(_line.shape) == 3: # permutation
                        num_permute, num_points, coords_dim = _line.shape
                        lines.append(torch.tensor(_line).reshape(num_permute, -1)) # (38, 40)
                    elif len(_line.shape) == 2:
                        lines.append(torch.tensor(_line).reshape(-1)) # (40, )
                    else:
                        assert False

            all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            all_lines_list.append(torch.stack(lines).float().to(device))

        gts = {
            'labels': all_labels_list,
            'lines': all_lines_list
        }
        gts = [deepcopy(gts) for _ in range(3)]
        return gts, imgs, valid_idx, points

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
    
    def eval(self):
        super().eval()

    def post_process(self, preds_dict, tokens, thr=0.0):
        lines = preds_dict['lines']  # List[Tensor(num_queries, 2*num_points)]
        bs = len(lines)
        scores = preds_dict['scores']  # (bs, num_queries, 3)
        prop_mask = preds_dict['prop_mask']

        results = []
        for i in range(bs):
            tmp_vectors = lines[i]
            tmp_prop_mask = prop_mask[i]
            num_preds, num_points2 = tmp_vectors.shape
            tmp_vectors = tmp_vectors.view(num_preds, num_points2 // 2, 2)
            # focal loss
            tmp_scores, tmp_labels = scores[i].max(-1)
            tmp_scores = tmp_scores.sigmoid()
            pos = tmp_scores > thr

            tmp_vectors = tmp_vectors[pos]
            tmp_scores = tmp_scores[pos]
            tmp_labels = tmp_labels[pos]
            tmp_prop_mask = tmp_prop_mask[pos]

            if len(tmp_scores) == 0:
                single_result = {
                    'vectors': [],
                    'scores': [],
                    'labels': [],
                    'prop_mask': [],
                    'token': tokens[i]
                }
            else:
                single_result = {
                    'vectors': tmp_vectors.detach().cpu().numpy(),
                    'scores': tmp_scores.detach().cpu().numpy(),
                    'labels': tmp_labels.detach().cpu().numpy(),
                    'prop_mask': tmp_prop_mask.detach().cpu().numpy(),
                    'token': tokens[i]
                }
            results.append(single_result)

        return results

    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_labels,
                           gt_lines,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                score_pred (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, 2*num_points]
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
                gt_lines (Tensor):
                    shape [num_gt, 2*num_points].

            Returns:
                tuple[Tensor]: a tuple containing the following for one sample.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_pred_lines = len(lines_pred)
        # assigner and sampler
        assign_result, gt_permute_idx = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred, ),
                                                             gts=dict(lines=gt_lines,
                                                                      labels=gt_labels, ),
                                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(
            assign_result, lines_pred, gt_lines)
        num_gt = len(gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lines.new_full(
            (num_pred_lines,), self.num_classes, dtype=torch.long)  # (num_q, )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines)  # (num_q, )

        lines_target = torch.zeros_like(lines_pred)  # (num_q, 2*num_pts)
        lines_weights = torch.zeros_like(lines_pred)  # (num_q, 2*num_pts)

        if num_gt > 0:
            if gt_permute_idx is not None:  # using permute invariant label
                # gt_permute_idx: (num_q, num_gt)
                # pos_inds: which query is positive
                # pos_gt_inds: which gt each pos pred is assigned
                # single_matched_gt_permute_idx: which permute order is matched
                single_matched_gt_permute_idx = gt_permute_idx[
                    pos_inds, pos_gt_inds
                ]
                lines_target[pos_inds] = gt_lines[pos_gt_inds, single_matched_gt_permute_idx].type(
                    lines_target.dtype)  # (num_q, 2*num_pts)
            else:
                lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
                    lines_target.dtype)  # (num_q, 2*num_pts)

        lines_weights[pos_inds] = 1.0  # (num_q, 2*num_pts)

        # normalization
        # n = lines_weights.sum(-1, keepdim=True) # (num_q, 1)
        # lines_weights = lines_weights / n.masked_fill(n == 0, 1) # (num_q, 2*num_pts)
        # [0, ..., 0] for neg ind and [1/npts, ..., 1/npts] for pos ind

        return (labels, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds, pos_gt_inds)

    def get_targets(self, preds, gts, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                preds (dict):
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # format the inputs
        gt_labels = gts['labels']
        gt_lines = gts['lines']

        lines_pred = preds['lines']

        (labels_list, label_weights_list,
         lines_targets_list, lines_weights_list,
         pos_inds_list, neg_inds_list, pos_gt_inds_list) = multi_apply(
            self._get_target_single, preds['scores'], lines_pred,
            gt_labels, gt_lines, gt_bboxes_ignore=gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        new_gts = dict(
            labels=labels_list,  # list[Tensor(num_q, )], length=bs
            label_weights=label_weights_list,  # list[Tensor(num_q, )], length=bs, all ones
            lines=lines_targets_list,  # list[Tensor(num_q, 2*num_pts)], length=bs
            lines_weights=lines_weights_list,  # list[Tensor(num_q, 2*num_pts)], length=bs
        )

        return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list

    def loss_single(self,
                    preds,
                    gts,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                preds (dict):
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # Get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list = \
            self.get_targets(preds, gts, gt_bboxes_ignore_list)

        # Batched all data
        # for k, v in new_gts.items():
        #     new_gts[k] = torch.stack(v, dim=0) # tensor (bs, num_q, ...)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * 0.

        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'][0].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Classification loss
        # since the inputs needs the second dim is the class dim, we permute the prediction.
        pred_scores = torch.cat(preds['scores'], dim=0)  # (bs*num_q, cls_out_channles)
        cls_scores = pred_scores.reshape(-1, self.num_classes)  # (bs*num_q, num_classes)
        cls_labels = torch.cat(new_gts['labels'], dim=0).reshape(-1)  # (bs*num_q, )
        cls_weights = torch.cat(new_gts['label_weights'], dim=0).reshape(-1)  # (bs*num_q, )
        
        loss_cls = self.loss_cls(
            cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        pred_lines = torch.cat(preds['lines'], dim=0)
        gt_lines = torch.cat(new_gts['lines'], dim=0)
        line_weights = torch.cat(new_gts['lines_weights'], dim=0)

        assert len(pred_lines) == len(gt_lines)
        assert len(gt_lines) == len(line_weights)

        loss_reg = self.loss_reg(
            pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        return loss_dict, pos_inds_list, pos_gt_inds_list, new_gts['lines']

    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss(self,
             gts,
             preds,
             gt_bboxes_ignore=None,
             reduction='mean'):
        """
            Loss Function.
            Args:
                gts (list[dict]): list length: num_layers
                    dict {
                        'label': list[tensor(num_gts, )], list length: batchsize,
                        'line': list[tensor(num_gts, 2*num_points)], list length: batchsize,
                        ...
                    }
                preds (list[dict]): list length: num_layers
                    dict {
                        'lines': tensor(bs, num_queries, 2*num_points),
                        'scores': tensor(bs, num_queries, class_out_channels),
                    }

                gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # Since there might have multi layer
        losses, pos_inds_lists, pos_gt_inds_lists, gt_lines_list = multi_apply(
            self.loss_single, preds, gts, reduction=reduction)

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v

        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict, pos_inds_lists, pos_gt_inds_lists, gt_lines_list




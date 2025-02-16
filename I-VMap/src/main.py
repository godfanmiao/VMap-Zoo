import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# 数据集类
class BEVDataset(Dataset):
    def __init__(self, image_paths, bev_matrices, transform=None):
        """
        :param image_paths: 图片路径列表，每个样本包含多个图片路径
        :param bev_matrices: BEV转换矩阵列表，每个样本对应一个矩阵
        :param transform: 数据预处理
        """
        self.image_paths = image_paths
        self.bev_matrices = bev_matrices
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图片
        images = [cv2.imread(path) for path in self.image_paths[idx]]
        if self.transform:
            images = [self.transform(image) for image in images]
        images = torch.stack(images)  # [T, C, H, W]

        # 加载BEV转换矩阵
        bev_matrix = torch.tensor(self.bev_matrices[idx], dtype=torch.float32)  # [4, 4]

        return {
            'images': images,  # [T, C, H, W]
            'bev_matrix': bev_matrix  # [4, 4]
        }

# 图片特征提取模块
class ImageFeatureExtractor(nn.Module):
    def __init__(self, hidden_dim=256):
        super(ImageFeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, images):
        """
        :param images: [T, C, H, W]
        :return: 图片特征 [T, hidden_dim, H', W']
        """
        T, C, H, W = images.shape
        images = images.view(-1, C, H, W)  # 展平时间维度
        features = self.backbone(images)  # 提取特征
        features = features.view(T, -1, features.shape[-2], features.shape[-1])  # 恢复时间维度
        return features

# BEV转换模块
class BEVTransformer(nn.Module):
    def __init__(self, hidden_dim=256):
        super(BEVTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, image_features, bev_matrix):
        """
        :param image_features: 图片特征 [T, hidden_dim, H', W']
        :param bev_matrix: BEV转换矩阵 [4, 4]
        :return: BEV特征 [T, hidden_dim, H', W']
        """
        T, C, H, W = image_features.shape
        bev_matrix = bev_matrix.view(1, 4, 1, 1).expand(T, -1, H, W)  # [T, 4, H', W']

        # 融合BEV矩阵
        fused_features = torch.cat([image_features, bev_matrix], dim=1)
        bev_features = self.linear(fused_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return bev_features

# 矢量地图解码器
class VectorMapDecoder(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=10, num_queries=100):
        super(VectorMapDecoder, self).__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # 输出类别
        self.bbox_head = nn.Linear(hidden_dim, 2)  # 输出二维坐标

    def forward(self, bev_features):
        """
        :param bev_features: BEV特征 [T, hidden_dim, H', W']
        :return: 预测的类别和坐标
        """
        T, C, H, W = bev_features.shape
        bev_features = bev_features.view(T, C, -1).permute(2, 0, 1)  # [L, T, hidden_dim]
        query = self.query_embed.weight.unsqueeze(1).repeat(1, T, 1)  # [num_queries, T, hidden_dim]
        hs = self.transformer(bev_features, query)  # Transformer输出
        outputs_class = self.class_head(hs)
        outputs_coords = self.bbox_head(hs).sigmoid()
        return outputs_class, outputs_coords

# 主模型
class MapTR(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=10, num_queries=100):
        super(MapTR, self).__init__()
        self.image_extractor = ImageFeatureExtractor(hidden_dim)
        self.bev_transformer = BEVTransformer(hidden_dim)
        self.decoder = VectorMapDecoder(hidden_dim, num_classes, num_queries)

    def forward(self, images, bev_matrix):
        """
        :param images: 连续多帧图片 [T, C, H, W]
        :param bev_matrix: BEV转换矩阵 [4, 4]
        :return: 预测的矢量地图
        """
        # 提取图片特征
        image_features = self.image_extractor(images)  # [T, hidden_dim, H', W']
        
        # 转换为BEV特征
        bev_features = self.bev_transformer(image_features, bev_matrix)  # [T, hidden_dim, H', W']
        
        # 解码矢量地图
        outputs_class, outputs_coords = self.decoder(bev_features)
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coords
        }

# 匈牙利匹配器
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            tgt_ids = torch.cat([t["labels"] for t in targets])
            tgt_bbox = torch.cat([t["boxes"] for t in targets])

            cost = -out_prob[:, tgt_ids]
            cost += self.cost_bbox * torch.cdist(out_bbox, tgt_bbox, p=1)
            cost += self.cost_giou * -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            C = cost.view(bs, num_queries, -1).cpu()
            sizes = [len(t["boxes"]) for t in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# 损失函数
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

# 辅助函数
def generalized_box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 示例数据
    image_paths = [["path/to/frame1.jpg", "path/to/frame2.jpg"], ...]  # 连续多帧图片路径
    bev_matrices = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]  # BEV转换矩阵

    dataset = BEVDataset(image_paths, bev_matrices)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = MapTR().to(device)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    criterion = SetCriterion(num_classes=10, matcher=matcher, weight_dict={'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1})
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for batch in dataloader:
            images = batch['images'].to(device)
            bev_matrix = batch['bev_matrix'].to(device)

            outputs = model(images, bev_matrix)
            targets = [{'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[0.1, 0.2], [0.3, 0.4]])}]  # 示例目标
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
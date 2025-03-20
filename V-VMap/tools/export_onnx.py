import argparse
import onnx
import numpy as np
from plugin.models import *
from mmcv import Config
from mmdet3d.models.builder import (build_backbone, build_head,build_neck)
from onnxsim import simplify
def get_args_parser():
    parser = argparse.ArgumentParser('export onnx', add_help=False)
    parser.add_argument('config', default='/media/yao/Data1/Litemapnet/plugin/configs/litemapnet.py',
                        help='train config file path')
    parser.add_argument('checkpoint', default='/media/yao/Data1/Litemapnet/weights/latest.pth',
                        help='path for checkpoint')
    parser.add_argument('--onnx_path', default='vvmap.onnx',
                        help='path for checkpoint')
    parser.add_argument('--simplify', default=True, action='store_true', type=bool)
    return parser


class VVMap(nn.Module):
    def __init__(self,model_cfg = None):
        super(VVMap, self).__init__()

        self.backbone = build_backbone(model_cfg['backbone_cfg'])
        self.neck = build_neck(model_cfg['neck_cfg'])
        self.ipm = build_neck(model_cfg['ipm_cfg'])
        self.head = build_head(model_cfg['head_cfg'])

    def extract_img_feat(self, imgs):

        img_feats = self.backbone(imgs)

        # reduce the channel dim
        img_feat = self.neck(img_feats)

        return img_feat

    def forward(self, img, ego2cam):

        img_shape = img.shape[-2:]

        img_feat = self.extract_img_feat(img)

        # Neck
        bev_feats = self.ipm(img_feat, ego2cam, img_shape)

        outputs_classes, outputs_coords = self.head(bev_feats)

        _, _, num_preds, num_points = outputs_coords.shape
        tmp_vectors = outputs_coords[-1].view(-1, num_preds, num_points // 2, 2)

        return outputs_classes[-1], tmp_vectors

def main(args):

    cfg = Config.fromfile(args.config)

    input_img = torch.randn(1, 3, cfg.img_h, cfg.img_w).cuda()
    input_ego2cam = torch.randn(1, 1, 4, 4, dtype=torch.double).cuda()

    input_names = ["input_img", "input_ego2cam"]
    out_names = ["outputs_classes", "outputs_coords"]

    model = VVMap(model_cfg=cfg.model)
    print(model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(checkpoint, strict=True)
    model.cuda()
    model.eval()

    torch.onnx.export(model, (input_img, input_ego2cam), args.onnx_path, export_params=True, training=False,
                      input_names=input_names, output_names=out_names, opset_version=13)

    if args.simplify:
        onnx_model = onnx.load(args.onnx_path)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, 'sim_'+args.onnx_path)
        print('finished exporting onnx')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
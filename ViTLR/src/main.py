'''
代码说明
时序信息处理：
在特征提取阶段，为每个时间步长添加时序编码（time_embed）。
通过Transformer处理时序特征，捕捉连续帧之间的关系。
匈牙利匹配：
使用HungarianMatcher类实现匈牙利匹配，计算预测框和真实框之间的最优匹配。
损失函数：
使用SetCriterion类计算分类损失（loss_ce）、边界框回归损失（loss_bbox）和GIoU损失（loss_giou）。
数据集：
FrameDataset类支持连续帧的加载，并返回时间步长信息。
模型结构：
DETR模型包括特征提取器（ResNet50）、Transformer、分类头和边界框回归头。
注意事项
数据集路径、类别数量、目标框格式等需要根据实际情况调整。
示例目标框和标签（targets）需要根据实际数据集进行修改。
代码中假设最多处理10帧连续帧，可以根据需求调整time_embed的维度。

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment
import numpy as np
from PIL import Image
import os

# 定义多层感知机
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# 定义DETR模型
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries):
        super().__init__()
        # Backbone: ResNet50去掉最后两层
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        # 1x1卷积降维
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        # 分类头和回归头
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 查询编码
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))
        # 位置编码
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        # 时序编码
        self.time_embed = nn.Parameter(torch.rand(10, hidden_dim // 2))  # 假设最多处理10帧

    def forward(self, inputs, time_steps):
        # inputs: [B, T, C, H, W]，time_steps: [B, T]
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)  # 展平时间维度
        x = self.backbone(inputs)  # 提取特征
        h = self.conv(x)  # 降维
        h = h.view(B, T, -1, h.shape[-2], h.shape[-1])  # 恢复时间维度

        # 添加时序编码
        time_steps = time_steps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1, 1]
        time_steps = self.time_embed[time_steps].squeeze(-1).squeeze(-1)  # [B, T, hidden_dim // 2]
        h += time_steps.unsqueeze(-1).unsqueeze(-1)  # 广播到特征图

        # 展平时间和空间维度
        h = h.flatten(2).permute(2, 0, 1)  # [L, B, hidden_dim]
        pos = self._build_position_encoding(h.shape[-1])  # 位置编码
        h = self.transformer(pos + h, self.query_pos.unsqueeze(1))  # Transformer

        outputs_class = self.class_embed(h)
        outputs_coord = self.bbox_embed(h).sigmoid()
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

    def _build_position_encoding(self, length):
        pos = torch.cat([
            self.col_embed[:length].unsqueeze(0).repeat(length, 1, 1),
            self.row_embed[:length].unsqueeze(1).repeat(1, length, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # [L, 1, hidden_dim]
        return pos

# 匈牙利匹配算法
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            tgt_ids = torch.cat([v["labels"] for v in targets])  # [num_target_boxes]
            tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [num_target_boxes, 4]

            cost_class = -out_prob[:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# 损失函数
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef: float = 0.1):
        super().__init__()
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

# 辅助函数：计算GIoU
def generalized_box_iou(boxes1, boxes2):
    # 计算IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union

    # 计算GIoU
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area

# 辅助函数：转换框格式
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 定义数据集
class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None, seq_length=10):
        self.frame_dir = frame_dir
        self.transform = transform
        self.seq_length = seq_length
        self.frames = sorted(os.listdir(frame_dir))

    def __len__(self):
        return len(self.frames) - self.seq_length + 1

    def __getitem__(self, idx):
        frames = [Image.open(os.path.join(self.frame_dir, self.frames[i])).convert('RGB') for i in range(idx, idx + self.seq_length)]
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        time_steps = torch.arange(self.seq_length)
        return frames, time_steps

# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集
    dataset = FrameDataset(frame_dir="path_to_frames", transform=transform, seq_length=10)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 模型
    model = DETR(num_classes=90, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100).to(device)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    criterion = SetCriterion(num_classes=90, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    for epoch in range(10):
        model.train()
        for frames, time_steps in train_loader:
            frames, time_steps = frames.to(device), time_steps.to(device)
            outputs = model(frames, time_steps)
            targets = [{"labels": torch.tensor([1, 2]), "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])}]  # 示例目标
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/10], Loss: {losses.item():.4f}")
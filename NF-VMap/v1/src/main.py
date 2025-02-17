import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.optimize import linear_sum_assignment

# 定义数据结构
class VectorMap:
    def __init__(self, trip_id, points):
        """
        :param trip_id: 趟数_id
        :param points: 矢量点列表，每个点为 {'point_id': int, 'x': float, 'y': float, 'class': int}
        """
        self.trip_id = trip_id
        self.points = points

# 数据集类
class VectorMapDataset(Dataset):
    def __init__(self, maps):
        self.maps = maps

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        map_data = self.maps[idx]
        points = map_data.points
        num_points = len(points)

        # 提取点的坐标和类别
        coords = torch.tensor([[p['x'], p['y']] for p in points], dtype=torch.float32)
        classes = torch.tensor([p['class'] for p in points], dtype=torch.long)

        return {
            'coords': coords,  # [N, 2]
            'classes': classes,  # [N]
            'num_points': num_points
        }


# Transformer模型
class MapTransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100, num_classes=10):
        super(MapTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # 嵌入层
        self.input_proj = nn.Linear(2, hidden_dim)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        # 条件层
        self.conditional_layer = nn.Linear(hidden_dim, hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)

        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 输出层
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_head = nn.Linear(hidden_dim, 2)

    def forward(self, coords, classes):
        """
        :param coords: 输入的点坐标 [N, 2]
        :param classes: 输入的点类别 [N]
        :return: 预测的类别和坐标
        """
        # 嵌入
        coord_embed = self.input_proj(coords)  # [N, hidden_dim]
        class_embed = self.class_embed(classes)  # [N, hidden_dim]

        # 条件调整
        conditional_embed = self.conditional_layer(class_embed)  # [N, hidden_dim]
        combined_embed = coord_embed + conditional_embed  # 条件调整后的嵌入

        # Transformer 输入
        src = combined_embed.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, src.size(1), 1)  # [num_queries, batch_size, hidden_dim]
        hs = self.transformer(src, tgt)  # Transformer 输出

        # 输出预测
        outputs_class = self.class_head(hs)
        outputs_coords = self.bbox_head(hs).sigmoid()
        return outputs_class, outputs_coords

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
    maps = [
        VectorMap(trip_id=1, points=[
            {'point_id': 1, 'x': 0.1, 'y': 0.2, 'class': 0},
            {'point_id': 2, 'x': 0.3, 'y': 0.4, 'class': 1}
        ]),
        VectorMap(trip_id=2, points=[
            {'point_id': 1, 'x': 0.2, 'y': 0.3, 'class': 0},
            {'point_id': 2, 'x': 0.4, 'y': 0.5, 'class': 1}
        ])
    ]

    dataset = VectorMapDataset(maps)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = MapTransformer().to(device)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    criterion = SetCriterion(num_classes=2, matcher=matcher, weight_dict={'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1})
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for batch in dataloader:
            coords = batch['coords'].to(device)
            classes = batch['classes'].to(device)
            num_points = batch['num_points']

            outputs_class, outputs_coords = model(coords)
            outputs = {
                'pred_logits': outputs_class,
                'pred_boxes': outputs_coords
            }
            targets = [{'labels': classes[i], 'boxes': coords[i]} for i in range(len(num_points))]
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
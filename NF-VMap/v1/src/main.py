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
    def __init__(self, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100, num_classes=10, max_points=50):
        super(MapTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.max_points = max_points

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
        self.polyline_head = nn.Linear(hidden_dim, max_points * 2)  # 每个点有 (x, y) 两个坐标

    def forward(self, coords, classes):
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
        outputs_polylines = self.polyline_head(hs).view(-1, self.max_points, 2)  # [num_queries, max_points, 2]
        return outputs_class, outputs_polylines

# 匈牙利匹配器
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_polyline=1):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_polyline = cost_polyline

    def polyline_distance(self, polyline1, polyline2):
        """
        Compute the average point-to-point distance between two polylines.
        :param polyline1: Tensor of shape [num_points1, 2] representing the first polyline.
        :param polyline2: Tensor of shape [num_points2, 2] representing the second polyline.
        :return: Average distance between the two polylines.
        """
        # 确保 polyline1 和 polyline2 的形状至少为 [1, 2]
        if polyline1.dim() == 1:
            polyline1 = polyline1.unsqueeze(0)  # 将形状从 [2] 转换为 [1, 2]
        if polyline2.dim() == 1:
            polyline2 = polyline2.unsqueeze(0)  # 将形状从 [2] 转换为 [1, 2]

        # 计算点到点的距离矩阵
        dist_matrix = torch.cdist(polyline1, polyline2, p=2)  # Shape: [num_points1, num_points2]

        # 计算 polyline1 到 polyline2 的最小距离
        min_dist1 = dist_matrix.min(dim=1).values.mean()
        # 计算 polyline2 到 polyline1 的最小距离
        min_dist2 = dist_matrix.min(dim=0).values.mean()

        # 返回两条 polyline 之间的平均距离
        return (min_dist1 + min_dist2) / 2

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes + 1]
            out_polylines = outputs["pred_polylines"]  # [batch_size * num_queries, max_points, 2]

            tgt_ids = torch.cat([t["labels"] for t in targets])  # [num_targets]
            tgt_polylines = torch.cat([t["polylines"] for t in targets], dim=0)  # [num_targets, max_points, 2]

            # 分类成本
            cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, num_targets]

            # polyline 距离成本
            cost_polyline = torch.zeros_like(cost_class)
            for i, pred_poly in enumerate(out_polylines):
                for j, tgt_poly in enumerate(tgt_polylines):
                    cost_polyline[i, j] = self.polyline_distance(pred_poly, tgt_poly)

            # 综合成本
            C = self.cost_class * cost_class + self.cost_polyline * cost_polyline
            C = C.view(bs, num_queries, -1).cpu()

            # 匈牙利匹配
            sizes = [len(t["labels"]) for t in targets]
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

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_polylines(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_polylines = outputs['pred_polylines'][idx]
        target_polylines = torch.cat([t['polylines'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_polyline = F.l1_loss(src_polylines, target_polylines, reduction='none')
        losses = {}
        losses['loss_polyline'] = loss_polyline.sum() / num_boxes
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
        losses.update(self.loss_polylines(outputs, targets, indices, num_boxes))
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
    matcher = HungarianMatcher(cost_class=1, cost_polyline=1)
    criterion = SetCriterion(num_classes=2, matcher=matcher, weight_dict={'loss_ce': 1, 'loss_polyline': 1})
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for batch in dataloader:
            coords = batch['coords'].to(device)
            classes = batch['classes'].to(device)
            num_points = batch['num_points']

            outputs_class, outputs_coords = model(coords, classes)
            outputs = {
                'pred_logits': outputs_class,
                'pred_polylines': outputs_coords
            }
            targets = [{'labels': classes[i], 'polylines': coords[i]} for i in range(len(num_points))]
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
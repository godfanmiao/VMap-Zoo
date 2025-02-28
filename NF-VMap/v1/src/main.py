import os
import re
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.optimize import linear_sum_assignment

# 数据集类
class VectorMapDataset(Dataset):
    def __init__(self, data_folder, max_trips=5, max_lines=20, points_per_line=50):
        """
        :param data_folder: 包含vec和label子文件夹的根目录路径
        :param max_trips: 单场景的最大趟数 (M_max)
        :param max_lines: 单趟的最大车道线数 (L_max)
        :param points_per_line: 每条车道线的采样点数 (N)
        """
        self.vec_folder = os.path.join(data_folder, "vec")
        self.label_folder = os.path.join(data_folder, "label")
        self.max_trips = max_trips
        self.max_lines = max_lines
        self.points_per_line = points_per_line
        self.maps = self._load_data_from_folders()

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        map_data = self.maps[idx]
        vec_orders = map_data["vec_orders"]
        label_order_lines = map_data["label_order_lines"]
        bounds = map_data["bounds"]  # Bounds范围用于归一化

        # 初始化输入张量和掩码矩阵
        input_tensor = torch.zeros((self.max_trips, self.max_lines, self.points_per_line, 3))
        label_tensor = torch.zeros((len(label_order_lines), self.points_per_line, 3))  # 真值认为是一趟，所以是(len(label_orders), N, 3)
        mask = torch.zeros((self.max_trips, self.max_lines, self.points_per_line), dtype=torch.bool)

        # 处理原始矢量数据
        for trip_idx, order in enumerate(vec_orders.values()):
            if trip_idx >= self.max_trips:
                break
            for line_idx, line in enumerate(order["lines"]):
                if line_idx >= self.max_lines:
                    break
                points = line["points"]
                normalized_points = self._normalize_coordinates(points, bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"])
                resampled_points = self._resample_points(normalized_points, self.points_per_line)
                input_tensor[trip_idx, line_idx, :, :2] = torch.tensor([[p['x'], p['y']] for p in resampled_points])
                input_tensor[trip_idx, line_idx, :, 2] = line["class"]
                for p_idx in range(self.points_per_line):
                    mask[trip_idx, line_idx, p_idx] = True

        # 处理真值矢量数据
        for line_idx, line in enumerate(label_order_lines):  # 假设真值只有一趟
            points = line["points"]
            normalized_points = self._normalize_coordinates(points, bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"])
            resampled_points = self._resample_points(normalized_points, self.points_per_line)
            label_tensor[line_idx, :, :2] = torch.tensor([[p['x'], p['y']] for p in resampled_points])
            label_tensor[line_idx, :, 2] = line["class"]

        return {
            "input_tensor": input_tensor,  # 输入张量
            "label_tensor": label_tensor,  # 真值张量
            "mask": mask,  # 掩码矩阵
            "bounds": bounds  # Bounds范围（可选，用于调试或反归一化）
        }

    def _load_data_from_folders(self):
        maps = []
        vec_files = os.listdir(self.vec_folder)
        label_files = os.listdir(self.label_folder)

        for vec_file in vec_files:
            if vec_file in label_files:
                vec_path = os.path.join(self.vec_folder, vec_file)
                label_path = os.path.join(self.label_folder, vec_file)
                map_data = self._parse_files(vec_path, label_path)
                maps.append(map_data)

        return maps

    def _parse_files(self, vec_path, label_path):
        """
        解析vec文件和label文件，生成VectorMap对象。
        """
        with open(vec_path, "r") as f:
            vec_lines = f.readlines()

        with open(label_path, "r") as f:
            label_lines = f.readlines()

        # 解析label文件中的Bounds
        bounds_match = re.search(r"Bounds: \(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\)", "\n".join(label_lines))
        if bounds_match:
            x_min, y_min, x_max, y_max = map(float, bounds_match.groups())
            bounds = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
        else:
            raise ValueError(f"Bounds not found in label file: {label_path}")

        # 解析vec文件
        vec_orders = defaultdict(lambda: {"order_id": None, "lines": []})
        current_order_id = None
        for line in vec_lines:
            line = line.strip()
            if line.startswith("OrderID:"):
                current_order_id = int(line.split(":")[1].strip())
                vec_orders[current_order_id]["order_id"] = current_order_id
            elif line.startswith("LINESTRING"):
                points_str = re.findall(r"\((.*?)\)", line)[0]
                points = [{"x": float(coord.split()[0]), "y": float(coord.split()[1])} for coord in points_str.split(", ")]
                vec_orders[current_order_id]["lines"].append({"points": points, "class": 0})

        # 解析label文件
        label_order_lines = vec_orders[current_order_id]["lines"]
        # label_order_lines = []
        # for line in label_lines:
        #     line = line.strip()
        #     if line.startswith("LINESTRING"):
        #         points_str = re.findall(r"\((.*?)\)", line)[0]
        #         points = [{"x": float(coord.split()[0]), "y": float(coord.split()[1])} for coord in points_str.split(", ")]
        #         label_order_lines.append({"points": points, "class": 0})

        return {
            "vec_orders": vec_orders, 
            "label_order_lines": label_order_lines, 
            "bounds": bounds
        }

    def _normalize_coordinates(self, points, x_min, x_max, y_min, y_max):
        """
        对坐标点进行归一化处理。
        :param points: 点列表，每个点为 {'x': float, 'y': float}
        :param x_min: x坐标的最小值
        :param x_max: x坐标的最大值
        :param y_min: y坐标的最小值
        :param y_max: y坐标的最大值
        :return: 归一化后的点列表
        """
        normalized_points = []
        for point in points:
            x = (point['x'] - x_min) / (x_max - x_min)
            y = (point['y'] - y_min) / (y_max - y_min)
            normalized_points.append({'x': x, 'y': y})
        return normalized_points

    def _resample_points(self, points, num_points):
        if len(points) > num_points:
            # 如果点数过多，均匀采样
            indices = torch.linspace(0, len(points) - 1, num_points).long()
            resampled_points = [points[i] for i in indices]
        else:
            # 如果点数不足，重复填充
            resampled_points = points * (num_points // len(points)) + points[:num_points % len(points)]
        return resampled_points


# Transformer模型
class MapTransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100, num_classes=10, max_trips=5, max_lines=20, points_per_line=50):
        super(MapTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.max_trips = max_trips
        self.max_lines = max_lines
        self.points_per_line = points_per_line

        # 嵌入层
        self.input_proj = nn.Linear(2, hidden_dim)  # 输入特征维度为2 (x, y)
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
        self.class_head = nn.Linear(hidden_dim, (num_classes + 1))
        self.polyline_head = nn.Linear(hidden_dim, points_per_line * 2)  # 输出坐标

    def forward(self, input_tensor, mask):
        # 输入张量形状：[B, M, L_max, N, 3]
        B, M, L_max, N, _ = input_tensor.shape

        # 展平输入张量以适应Transformer
        input_tensor = input_tensor.view(B, M * L_max * N, 3)  # [B, M * L_max * N, 3]
        mask = mask.view(B, M * L_max * N)  # [B, M * L_max * N]

        # 嵌入
        coord_embed = self.input_proj(input_tensor[..., :2])  # [B, M * L_max * N, hidden_dim]
        class_embed = self.class_embed(input_tensor[..., 2].long())  # [B, M * L_max * N, hidden_dim]
        # 条件调整
        conditional_embed = self.conditional_layer(class_embed)  # [N, hidden_dim]
        combined_embed = coord_embed + conditional_embed  # [B, M * L_max * N, hidden_dim]

        # Transformer 输入
        src = combined_embed.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, batch_size, hidden_dim]
        hs = self.transformer(src, tgt, src_key_padding_mask=mask)  # Transformer 输出

        # 输出预测
        outputs_class = self.class_head(hs)  # [num_queries, B, (num_classes + 1)]
        outputs_polylines = self.polyline_head(hs).sigmoid().view(self.num_queries, B, self.points_per_line, 2)  # [num_queries, B, N, 2]

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
        dist_matrix = torch.cdist(polyline1.view(-1, 2), polyline2.view(-1, 2), p=1)  # Shape: [num_points1, num_points2]

        # 计算 polyline1 到 polyline2 的最小距离
        min_dist1 = dist_matrix.min(dim=1).values.mean()
        # 计算 polyline2 到 polyline1 的最小距离
        min_dist2 = dist_matrix.min(dim=0).values.mean()

        # 返回两条 polyline 之间的平均距离
        return (min_dist1 + min_dist2) / 2

    def forward(self, outputs, targets):
        with torch.no_grad():
            num_queries, bs = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, (num_classes + 1)]
            out_polylines = outputs["pred_polylines"].flatten(0, 1)  # [batch_size * num_queries, max_points, 2]

            tgt_ids = torch.cat([t["labels"] for t in targets])  # [batch_size * L_max]
            tgt_polylines = torch.cat([t["polylines"] for t in targets], dim=0)  # [batch_size * L_max , N, 2]

            # 分类成本
            cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, batch_size * L_max]

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

    def loss_labels(self, outputs, targets, indices, num_polylines):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_polylines(self, outputs, targets, indices, num_polylines):
        idx = self._get_src_permutation_idx(indices)
        src_polylines = outputs['pred_polylines'][idx]
        target_polylines = torch.cat([t['polylines'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_polyline = F.l1_loss(src_polylines, target_polylines, reduction='none')
        losses = {}
        losses['loss_polyline'] = loss_polyline.sum() / num_polylines
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return src_idx, batch_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_polylines = sum(len(t["labels"]) for t in targets)
        num_polylines = torch.as_tensor([num_polylines], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_polylines = torch.clamp(num_polylines, min=1).item()
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_polylines))
        losses.update(self.loss_polylines(outputs, targets, indices, num_polylines))
        return losses


def normalize_coordinates(points, x_min, x_max, y_min, y_max):
    """
    对坐标点进行归一化处理。
    :param points: 点列表，每个点为 {'x': float, 'y': float}
    :param x_min: x坐标的最小值
    :param x_max: x坐标的最大值
    :param y_min: y坐标的最小值
    :param y_max: y坐标的最大值
    :return: 归一化后的点列表
    """
    normalized_points = []
    for point in points:
        x = (point['x'] - x_min) / (x_max - x_min)
        y = (point['y'] - y_min) / (y_max - y_min)
        normalized_points.append({'x': x, 'y': y})
    return normalized_points

def collate_fn(batch):
    """
    自定义collate_fn，用于处理不同长度的label_tensor。
    :param batch: 一个列表，每个元素是一个样本，由__getitem__返回。
    :return: 一个字典，包含堆叠后的输入张量、填充后的标签张量、掩码矩阵等。
    """
    # 提取每个样本的输入张量、标签张量、掩码矩阵和Bounds
    input_tensors = [item["input_tensor"] for item in batch]
    label_tensors = [item["label_tensor"] for item in batch]
    masks = [item["mask"] for item in batch]
    bounds = [item["bounds"] for item in batch]

    # 计算每个样本的标签数量
    label_lengths = [len(t) for t in label_tensors]

    # 找到最大标签数量
    max_label_length = max(label_lengths)

    # 填充label_tensor到最大长度
    padded_label_tensors = []
    label_masks = []
    for label_tensor, length in zip(label_tensors, label_lengths):
        # 填充零到最大长度
        padded_label_tensor = torch.zeros((max_label_length, label_tensor.shape[1], label_tensor.shape[2]))
        padded_label_tensor[:length] = label_tensor
        padded_label_tensors.append(padded_label_tensor)

        # 创建掩码矩阵
        label_mask = torch.zeros(max_label_length, dtype=torch.bool)
        label_mask[:length] = True
        label_masks.append(label_mask)

    # 将输入张量堆叠成一个批次
    input_tensors = torch.stack(input_tensors, dim=0)
    masks = torch.stack(masks, dim=0)
    label_masks = torch.stack(label_masks, dim=0)
    padded_label_tensors = torch.stack(padded_label_tensors, dim=0)

    return {
        "input_tensor": input_tensors,
        "label_tensor": padded_label_tensors,
        "mask": masks,
        "label_mask": label_masks,
        "bounds": bounds
    }

# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "checkpoints"  # 模型保存路径
    os.makedirs(save_dir, exist_ok=True)

    dataset = VectorMapDataset("D:\\NF-VMap\\dataset\\train_grids", max_trips=5, max_lines=10, points_per_line=10)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)

    model = MapTransformer(max_trips=5, max_lines=10, points_per_line=10, num_queries=10, num_classes=3).to(device)
    matcher = HungarianMatcher(cost_class=1, cost_polyline=1)
    criterion = SetCriterion(num_classes=3, matcher=matcher, weight_dict={'loss_ce': 1, 'loss_polyline': 1})
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for batch in dataloader:
            input_tensor = batch['input_tensor'].to(device)
            label_tensor = batch['label_tensor'].to(device)
            mask = batch['mask'].to(device)

            outputs_class, outputs_coords = model(input_tensor, mask)
            outputs = {
                'pred_logits': outputs_class,
                'pred_polylines': outputs_coords
            }

            # 提取类别标签和坐标
            label_classes = label_tensor[..., 2].long()  # 类别标签在最后一个维度
            label_coords = label_tensor[..., :2]  # 坐标在前两个维度

            targets = [{'labels': label_classes[b].t()[0], 'polylines': label_coords[b].view(-1, 10, 2)} for b in range(label_classes.size(0))]
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss_dict}")
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # 每个epoch保存一次模型
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))
        print(f"Model saved to {os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')}")
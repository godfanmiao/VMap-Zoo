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
import math
from torch.utils.tensorboard import SummaryWriter
import time

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
        input_mask = torch.ones((self.max_trips, self.max_lines), dtype=torch.bool)

        # 处理原始矢量数据
        for trip_idx, order in enumerate(vec_orders.values()):
            if trip_idx >= self.max_trips:
                break
            for line_idx, line in enumerate(order["lines"]):
                if line_idx >= self.max_lines:
                    break
                points = line["points"]
                normalized_points = self._normalize_coordinates(points, bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"])
                resampled_points = self._resample_polyline(normalized_points, self.points_per_line)
                input_tensor[trip_idx, line_idx, :, :2] = torch.tensor([[p[0], p[1]] for p in resampled_points])
                input_tensor[trip_idx, line_idx, :, 2] = line["class"]
                input_mask[trip_idx, line_idx] = False

        # 处理真值矢量数据
        for line_idx, line in enumerate(label_order_lines):  # 假设真值只有一趟
            points = line["points"]
            normalized_points = self._normalize_coordinates(points, bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"])
            resampled_points = self._resample_polyline(normalized_points, self.points_per_line)
            label_tensor[line_idx, :, :2] = torch.tensor([[p[0], p[1]] for p in resampled_points])
            label_tensor[line_idx, :, 2] = line["class"]

        
        # 将输入数据的一趟重复多次，并作为真值
        # input_tensor = self._replace_with_first_element(input_tensor, dim=0)
        # input_mask = self._replace_with_first_element(input_mask, dim=0)
        # label_tensor = input_tensor[0]

        return {
            "input_tensor": input_tensor,  # 输入张量
            "label_tensor": label_tensor,  # 真值张量
            "input_mask": input_mask,  # 掩码矩阵
            "bounds": bounds  # Bounds范围（可选，用于调试或反归一化）
        }
    
    def _replace_with_first_element(self, tensor, dim=0):
        # 获取第一个元素
        first_element = tensor[0]
        # 计算需要的重复次数（沿着指定维度）
        repeats = [tensor.shape[dim]] + [1] * (tensor.ndim - 1)
        # 使用 tile 复制第一个元素到整个维度
        repeated_element = first_element.tile(*repeats)
        # 替换原 tensor 的值
        tensor = repeated_element
        return tensor

    def _load_data_from_folders(self):
        maps = []
        vec_files = os.listdir(self.vec_folder)
        label_files = os.listdir(self.label_folder)

        for vec_file in vec_files:
            if vec_file in label_files and vec_file.endswith('.txt'):
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
                points = [(float(coord.split()[0]), float(coord.split()[1])) for coord in points_str.split(", ")]
                vec_orders[current_order_id]["lines"].append({"points": points, "class": 0})

        # 解析label文件
        label_order_lines = []
        for line in label_lines:
            line = line.strip()
            if line.startswith("LINESTRING"):
                points_str = re.findall(r"\((.*?)\)", line)[0]
                points = [(float(coord.split()[0]), float(coord.split()[1])) for coord in points_str.split(", ")]
                label_order_lines.append({"points": points, "class": 0})

        return {
            "vec_orders": vec_orders, 
            "label_order_lines": label_order_lines, 
            "bounds": bounds
        }

    def _normalize_coordinates(self, points, x_min, x_max, y_min, y_max):
        """
        对坐标点进行归一化处理。
        :param points: 点列表，格式为 [(x1, y1), (x2, y2), ...]
        :param x_min: x坐标的最小值
        :param x_max: x坐标的最大值
        :param y_min: y坐标的最小值
        :param y_max: y坐标的最大值
        :return: 归一化后的点列表
        """
        normalized_points = []
        for point in points:
            x = (point[0] - x_min) / (x_max - x_min)
            y = (point[1] - y_min) / (y_max - y_min)
            normalized_points.append((x, y))
        return normalized_points

    def _compute_length(self, point1, point2):
        """计算两点之间的欧几里得距离"""
        return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

    def _calculate_total_length(self, polylines):
        """计算折线的总长度"""
        total_length = 0.0
        for i in range(len(polylines) - 1):
            total_length += self._compute_length(polylines[i], polylines[i+1])
        return total_length

    def _resample_polyline(self, polylines, num_points):
        """
        对折线进行重采样，使其包含固定数量的点
        :param polylines: 原始折线的点列表，格式为 [(x1, y1), (x2, y2), ...]
        :param num_points: 需要重采样成的点数
        :return: 重采样后的点列表
        """
        if len(polylines) < 2:
            raise ValueError("折线至少需要两个点才能重采样")
        
        # 计算折线的总长度
        total_length = self._calculate_total_length(polylines)
        if total_length == 0:
            return polylines  # 如果总长度为0，返回原点
        
        # 计算目标点间距
        segment_length = total_length / (num_points - 1)
        sampled_points = []
        current_point = 0  # 当前到达的点索引
        current_length = 0.0  # 累计距离
        
        sampled_points.append(polylines[0])  # 添加起点
        
        # 遍历每条线段，按等间距采样
        for i in range(len(polylines) - 1):
            a = polylines[i]
            b = polylines[i+1]
            segment_length_ab = self._compute_length(a, b)
            if segment_length_ab == 0:
                continue  # 忽略重合点，避免除以零
            
            # 计算该线段需要插入的点
            t = 0.0  # 线段上的参数，范围 [0, 1]
            while t <= 1.0:
                # 当前位置
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                point = (x, y)
                
                # 计算当前点到起点的总距离
                current_distance = self._compute_length(polylines[0], point)
                
                # 判断是否需要添加这个点
                if len(sampled_points) < num_points:
                    if current_distance >= segment_length * len(sampled_points):
                        sampled_points.append(point)
                
                # 更新 t，继续沿着线段前进
                t += segment_length / segment_length_ab
            
            # 跳出循环时，可能最后一点需要额外处理
            current_distance = self._compute_length(polylines[0], b)
            if len(sampled_points) < num_points:
                if current_distance >= segment_length * len(sampled_points):
                    sampled_points.append(b)
        
        # 确保最后一个点被添加
        if len(sampled_points) < num_points:
            sampled_points.append(polylines[-1])
        
        # 截取前 num_points 个点
        sampled_points = sampled_points[:num_points]
        
        return sampled_points

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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
        self.input_proj = MLP(2 * points_per_line, hidden_dim, hidden_dim, 3)  # 输入特征维度为2 * points_per_line (x, y)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        # 条件层
        self.conditional_layer = nn.Linear(hidden_dim, hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first=True)

        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 输出层
        self.class_head = nn.Linear(hidden_dim, (num_classes + 1))
        # self.polyline_head = nn.Linear(hidden_dim, points_per_line * 2)  # 输出坐标
        self.polyline_head = MLP(hidden_dim, hidden_dim, points_per_line * 2, 3)

        self.trip_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))  # 每个trip的嵌入
        self.line_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))


        # 自定义初始化
        # 初始化权重
        # nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.constant_(self.input_proj.bias, 0)

        # nn.init.kaiming_normal_(self.conditional_layer.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.constant_(self.conditional_layer.bias, 0)

        # nn.init.xavier_normal_(self.class_embed.weight)
        # nn.init.xavier_normal_(self.query_embed.weight)

    def forward(self, input_tensor, mask):
        # 输入张量形状：[B, M, L_max, N, 3]
        B, M, L_max, N, _ = input_tensor.shape

        # 展平输入张量以适应Transformer
        mask = mask.view(B, M * L_max)  # [B, M * L_max]

        # 嵌入
        coord_embed = self.input_proj(input_tensor[..., :2].reshape(B, M * L_max, N * 2))  # [B, M * L_max, hidden_dim]
        class_embed = self.class_embed(input_tensor[..., 2].view(B, M * L_max, N)[..., 0].long())  # [B, M * L_max, hidden_dim]
        # 条件调整
        conditional_embed = self.conditional_layer(class_embed)  # [B, M * L_max, hidden_dim]
        combined_embed = coord_embed + conditional_embed  # [B, M * L_max, hidden_dim]

        pos = torch.cat([
            self.trip_embed[:M].unsqueeze(0).repeat(L_max, 1, 1),
            self.line_embed[:L_max].unsqueeze(1).repeat(1, M, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(0).repeat(B, 1, 1)  # [B, M * L_max, hidden_dim]

        # Transformer 输入
        src = combined_embed  # [B, M * L_max, hidden_dim]
        tgt = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [batch_size, num_queries, hidden_dim]
        hs = self.transformer(pos + src, tgt, src_key_padding_mask=mask)  # Transformer 输出

        # 输出预测
        outputs_class = self.class_head(hs)  # [B, num_queries, (num_classes + 1)]
        outputs_polylines = self.polyline_head(hs).sigmoid()  # [B, num_queries, N, 2]

        return outputs_class, outputs_polylines

# 匈牙利匹配器
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_polyline=1, cost_direction=1):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_polyline = cost_polyline
        self.cost_direction = cost_direction

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
        dist_matrix = torch.cdist(polyline1, polyline2, p=1)  # Shape: [num_points1, num_points2]

        # 计算 polyline1 到 polyline2 的最小距离
        min_dist1 = dist_matrix.min(dim=1).values.mean()
        # 计算 polyline2 到 polyline1 的最小距离
        min_dist2 = dist_matrix.min(dim=0).values.mean()

        # 返回两条 polyline 之间的平均距离
        return (min_dist1 + min_dist2) / 2

    def discrete_frechet_distance(self, polyline1, polyline2):
        """
        Compute the discrete Fréchet distance between two polylines using dynamic programming.
        :param polyline1: Tensor of shape [num_points1, 2] representing the first polyline.
        :param polyline2: Tensor of shape [num_points2, 2] representing the second polyline.
        :return: Discrete Fréchet distance between the two polylines.
        """
        # 确保输入是二维张量
        if polyline1.dim() == 1:
            polyline1 = polyline1.unsqueeze(0)
        if polyline2.dim() == 1:
            polyline2 = polyline2.unsqueeze(0)

        m = polyline1.size(0)
        n = polyline2.size(0)

        # 初始化动态规划表
        dp = torch.zeros(m, n, dtype=torch.float32)

        # 定义欧几里得距离函数
        def distance(i, j):
            return torch.norm(polyline1[i] - polyline2[j], p=2)

        # 填充动态规划表
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    dp[i][j] = distance(i, j)
                elif i == 0:
                    dp[i][j] = torch.max(dp[i][j - 1], distance(i, j))
                elif j == 0:
                    dp[i][j] = torch.max(dp[i - 1][j], distance(i, j))
                else:
                    candidate = torch.min(torch.stack([dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]]))
                    dp[i][j] = torch.max(candidate, distance(i, j))

        return dp[-1][-1].item()
    
    def direction_loss(self, pred_polyline, gt_polyline):
        """
        Compute the direction loss between two polylines.
        :param pred_polyline: Predicted polyline tensor of shape [N, 2]
        :param gt_polyline: Ground truth polyline tensor of shape [N, 2]
        :return: Direction loss
        """
        # 计算首尾点的方向向量
        pred_direction = pred_polyline[-1] - pred_polyline[0]
        gt_direction = gt_polyline[-1] - gt_polyline[0]

        # 归一化方向向量
        pred_direction = pred_direction / (torch.norm(pred_direction) + 1e-6)
        gt_direction = gt_direction / (torch.norm(gt_direction) + 1e-6)

        # 计算方向损失（余弦相似度）
        cosine_similarity = torch.dot(pred_direction, gt_direction)
        return 1 - cosine_similarity

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
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

            # 方向成本
            cost_direction = torch.zeros_like(cost_class)
            for i, pred_poly in enumerate(out_polylines):
                for j, tgt_poly in enumerate(tgt_polylines):
                    cost_direction[i, j] = self.direction_loss(pred_poly, tgt_poly)
            
            # 综合成本
            C = self.cost_class * cost_class + self.cost_polyline * cost_polyline + self.cost_direction * cost_direction
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
        # 方向损失计算
        polyline_loss = 0.0
        for polyline1, polyline2 in zip(src_polylines, target_polylines):
            # 确保 polyline1 和 polyline2 的形状至少为 [1, 2]
            if polyline1.dim() == 1:
                polyline1 = polyline1.unsqueeze(0)  # 将形状从 [2] 转换为 [1, 2]
            if polyline2.dim() == 1:
                polyline2 = polyline2.unsqueeze(0)  # 将形状从 [2] 转换为 [1, 2]

            # 计算点到点的距离矩阵
            dist_matrix = torch.cdist(polyline1, polyline2, p=1)  # Shape: [num_points1, num_points2]

            # 计算 polyline1 到 polyline2 的最小距离
            min_dist1 = dist_matrix.min(dim=1).values.mean()
            # 计算 polyline2 到 polyline1 的最小距离
            min_dist2 = dist_matrix.min(dim=0).values.mean()

            # 两条 polyline 之间的平均距离
            polyline_loss += 1 - (min_dist1 + min_dist2) / 2
        polyline_loss /= num_polylines
        losses = {'loss_polyline': polyline_loss}
        return losses
        
    def loss_direction(self, outputs, targets, indices, num_polylines):
        idx = self._get_src_permutation_idx(indices)
        src_polylines = outputs['pred_polylines'][idx]
        target_polylines = torch.cat([t['polylines'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 方向损失计算
        direction_loss = 0.0
        for src, tgt in zip(src_polylines, target_polylines):
            # 计算方向向量
            src_diff = src[-1] - src[0]
            tgt_diff = tgt[-1] - tgt[0]
            # 归一化
            src_diff = src_diff / (torch.norm(src_diff) + 1e-6)
            tgt_diff = tgt_diff / (torch.norm(tgt_diff) + 1e-6)
            # 余弦相似度
            cosine_similarity = torch.dot(src_diff, tgt_diff)
            direction_loss += 1 - cosine_similarity
        direction_loss /= num_polylines
        losses = {'loss_direction': direction_loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_polylines = sum(len(t["labels"]) for t in targets)
        num_polylines = torch.as_tensor([num_polylines], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_polylines = torch.clamp(num_polylines, min=1).item()
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_polylines))
        losses.update(self.loss_polylines(outputs, targets, indices, num_polylines))
        losses.update(self.loss_direction(outputs, targets, indices, num_polylines))
        return losses

def collate_fn(batch):
    """
    自定义collate_fn，用于处理不同长度的label_tensor。
    :param batch: 一个列表，每个元素是一个样本，由__getitem__返回。
    :return: 一个字典，包含堆叠后的输入张量、填充后的标签张量、掩码矩阵等。
    """
    # 提取每个样本的输入张量、标签张量、掩码矩阵和Bounds
    input_tensors = [item["input_tensor"] for item in batch]
    label_tensors = [item["label_tensor"] for item in batch]
    input_masks = [item["input_mask"] for item in batch]
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
        label_mask[:length] = False
        label_masks.append(label_mask)

    # 将输入张量堆叠成一个批次
    input_tensors = torch.stack(input_tensors, dim=0)
    input_masks = torch.stack(input_masks, dim=0)
    label_masks = torch.stack(label_masks, dim=0)
    padded_label_tensors = torch.stack(padded_label_tensors, dim=0)

    return {
        "input_tensor": input_tensors,
        "label_tensor": padded_label_tensors,
        "input_mask": input_masks,
        "label_mask": label_masks,
        "bounds": bounds
    }

# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "checkpoints"  # 模型保存路径
    os.makedirs(save_dir, exist_ok=True)

    run_dir = "runs"  # TensorBoard 日志保存路径
    version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = os.path.join(run_dir, f"logs-{version}")  # TensorBoard 日志保存路径
    writer = SummaryWriter(log_dir=log_dir)  # 初始化 TensorBoard SummaryWriter

    total_epochs = 500  # 总训练轮数
    save_interval = 10  # 每多少轮保存一次模型
    train_batch_size = 8  # 训练批次大小
    val_batch_size = 2  # 验证批次大小

    num_queries = 5  # 每个样本的查询次数
    max_trips = 5  # 每个样本的最大趟数
    max_lines = 10  # 每趟的最大线段数
    points_per_line = 10  # 每条线段的最大点数

    train_dataset = VectorMapDataset("D:/NF-VMap/dataset/train", max_trips, max_lines, points_per_line)
    val_dataset = VectorMapDataset("D:/NF-VMap/dataset/val", max_trips, max_lines, points_per_line)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    model = MapTransformer(max_trips=max_trips, max_lines=max_lines, points_per_line=points_per_line, num_queries=num_queries, num_classes=3).to(device)
    matcher = HungarianMatcher(cost_class=1, cost_polyline=1)
    criterion = SetCriterion(num_classes=3, matcher=matcher, weight_dict={'loss_ce': 1, 'loss_polyline': 3, 'loss_direction': 1})
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=5e-5)

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            input_tensor = batch['input_tensor'].to(device)
            label_tensor = batch['label_tensor'].to(device)
            input_mask = batch['input_mask'].to(device)
            label_mask = batch['label_mask'].to(device)

            outputs_class, outputs_coords = model(input_tensor, input_mask)
            outputs = {
                'pred_logits': outputs_class,
                'pred_polylines': outputs_coords.view(train_batch_size, num_queries, points_per_line, 2)
            }

            # 提取类别标签和坐标
            label_classes = label_tensor[..., 2].long()  # 类别标签在最后一个维度
            label_coords = label_tensor[..., :2]  # 坐标在前两个维度

            targets = [{
                'labels': label_classes[b].t()[0][~label_mask[b]], 
                'polylines': label_coords[b].view(-1, 10, 2)[~label_mask[b]]
            } for b in range(label_classes.size(0))]
            loss_dict = criterion(outputs, targets)
            print(f"Loss: CE={loss_dict['loss_ce']:.4f}, Polyline={loss_dict['loss_polyline']:.4f}, Direction={loss_dict['loss_direction']:.4f}")
            loss = sum(loss_dict.values())
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"Epoch {epoch + 1}, Loss: {loss_dict}")
            # print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()}")
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        # 计算平均训练损失
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_tensor = batch['input_tensor'].to(device)
                label_tensor = batch['label_tensor'].to(device)
                input_mask = batch['input_mask'].to(device)
                label_mask = batch['label_mask'].to(device)

                outputs_class, outputs_coords = model(input_tensor, input_mask)
                outputs = {
                    'pred_logits': outputs_class,
                    'pred_polylines': outputs_coords.view(val_batch_size, num_queries, points_per_line, 2)
                }

                label_classes = label_tensor[..., 2].long()
                label_coords = label_tensor[..., :2]

                targets = [{
                    'labels': label_classes[b].t()[0][~label_mask[b]], 
                    'polylines': label_coords[b].view(-1, 10, 2)[~label_mask[b]]
                } for b in range(label_classes.size(0))]
                loss_dict = criterion(outputs, targets)
                print(f"Loss: CE={loss_dict['loss_ce']:.4f}, Polyline={loss_dict['loss_polyline']:.4f}, Direction={loss_dict['loss_direction']:.4f}")
                loss = sum(loss_dict.values())
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/validation', val_loss, epoch + 1)

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            # 每个epoch保存一次模型
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))
            print(f"Model saved to {os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')}")

    # 关闭 TensorBoard writer
    writer.close()
'''好的！如果类别ID是离散变量，我们需要对模型的输出进行调整，以确保类别ID的预测是分类任务，而不是回归任务。具体来说，我们需要将类别ID的预测部分改为分类输出，并使用交叉熵损失来计算类别ID的损失。
以下是优化后的代码，其中类别ID的预测部分使用了分类输出，并调整了损失函数以适应离散变量的处理。
优化后的代码
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from scipy.optimize import linear_sum_assignment

# 定义模型
class MapTRModel(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, num_classes=10):
        super(MapTRModel, self).__init__()
        # 卫星图像处理模块（CNN）
        self.satellite_cnn = models.resnet18(pretrained=True)
        self.satellite_cnn.fc = nn.Linear(self.satellite_cnn.fc.in_features, embed_dim)

        # 矢量点图处理模块（Transformer Encoder）
        self.vector_embedding = nn.Linear(3, embed_dim)  # 输入为类别ID和二维坐标
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.vector_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 特征融合模块（Transformer Encoder）
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 解码模块（Transformer Decoder）
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出模块
        self.classifier = nn.Linear(embed_dim, num_classes)  # 分类输出类别ID
        self.regressor = nn.Linear(embed_dim, 2)  # 回归输出二维坐标

    def forward(self, satellite_image, vector_points):
        # 卫星图像特征提取
        satellite_features = self.satellite_cnn(satellite_image)
        satellite_features = satellite_features.unsqueeze(1)  # [batch_size, 1, embed_dim]

        # 矢量点图特征提取
        vector_points = self.vector_embedding(vector_points)
        vector_points = vector_points.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        vector_features = self.vector_transformer(vector_points)

        # 特征融合
        satellite_features = satellite_features.permute(1, 0, 2)  # [1, batch_size, embed_dim]
        fused_features = torch.cat((vector_features, satellite_features), dim=0)  # [seq_len+1, batch_size, embed_dim]
        fused_features = self.fusion_transformer(fused_features)

        # 解码
        tgt = torch.zeros_like(vector_points)  # 初始化解码器的目标序列
        decoded_features = self.decoder(tgt, fused_features)

        # 输出预测
        decoded_features = decoded_features.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        class_predictions = self.classifier(decoded_features)  # 分类预测类别ID
        point_predictions = self.regressor(decoded_features)  # 回归预测二维坐标
        predictions = torch.cat((class_predictions.unsqueeze(-1), point_predictions), dim=-1)  # [batch_size, seq_len, 3]
        return predictions


# 匈牙利匹配
def hungarian_matching(cost_matrix):
    """
    使用匈牙利算法计算最优匹配
    :param cost_matrix: 成本矩阵 [batch_size, num_preds, num_gts]
    :return: 匹配索引 [batch_size, num_preds]
    """
    batch_size, num_preds, num_gts = cost_matrix.shape
    matches = []
    for i in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(cost_matrix[i].cpu().numpy())
        match = torch.zeros(num_preds, dtype=torch.long, device=cost_matrix.device)
        match[row_ind] = torch.tensor(col_ind, device=cost_matrix.device)
        matches.append(match)
    return torch.stack(matches)


# MapTR风格的损失函数
class MapTRLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_pts=5.0):
        super(MapTRLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_pts = lambda_pts
        self.classification_loss = nn.CrossEntropyLoss()
        self.point_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        :param predictions: [batch_size, num_preds, 3] (类别ID, x, y)
        :param targets: [batch_size, num_gts, 3] (类别ID, x, y)
        :return: 总损失
        """
        batch_size, num_preds, _ = predictions.shape
        _, num_gts, _ = targets.shape

        # 提取类别ID和坐标
        pred_classes = predictions[:, :, 0].view(-1, predictions.size(-1))  # [batch_size * num_preds, num_classes]
        pred_points = predictions[:, :, 1:]  # [batch_size, num_preds, 2]
        target_classes = targets[:, :, 0].view(-1).long()  # [batch_size * num_gts]
        target_points = targets[:, :, 1:]  # [batch_size, num_gts, 2]

        # 分类损失
        cls_loss = self.classification_loss(pred_classes, target_classes)

        # 匈牙利匹配
        cost_matrix = torch.cdist(pred_points, target_points, p=2)  # [batch_size, num_preds, num_gts]
        matches = hungarian_matching(cost_matrix)

        # 点对点损失
        pts_loss = 0
        for i in range(batch_size):
            matched_gts = target_points[i, matches[i]]
            pts_loss += self.point_loss(pred_points[i], matched_gts)
        pts_loss /= batch_size

        # 总损失
        total_loss = self.lambda_cls * cls_loss + self.lambda_pts * pts_loss
        return total_loss


# 示例数据加载器
class MapTRDataset(Dataset):
    def __init__(self, satellite_images, vector_points, labels):
        self.satellite_images = satellite_images
        self.vector_points = vector_points
        self.labels = labels

    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        return self.satellite_images[idx], self.vector_points[idx], self.labels[idx]


# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for satellite_image, vector_points, labels in dataloader:
        satellite_image, vector_points, labels = satellite_image.to(device), vector_points.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(satellite_image, vector_points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Loss: {total_loss / len(dataloader)}")


# 示例数据
satellite_images = torch.randn(10, 3, 224, 224)  # 示例卫星图像数据
vector_points = torch.randn(10, 5, 3)  # 示例矢量点图数据，5个点，每个点3个特征（类别ID, x, y）
labels = torch.randn(10, 5, 3)  # 示例标签数据

# 将类别ID转换为离散变量
vector_points[:, :, 0] = torch.randint(0, 10, (10, 5)).float()  # 随机生成类别ID
labels[:, :, 0] = torch.randint(0, 10, (10, 5)).float()  # 随机生成类别ID

dataset = MapTRDataset(satellite_images, vector_points, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 初始化模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MapTRModel(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = MapTRLoss()

# 训练模型
train(model, dataloader, optimizer, criterion, device)

'''
优化点说明
类别ID的处理：
类别ID是离散变量，因此在模型输出部分，类别ID的预测使用了nn.Linear(embed_dim, num_classes)，并使用nn.CrossEntropyLoss计算分类损失。
坐标部分仍然是回归任务，使用nn.MSELoss计算损失。
损失函数：
分类损失和点对点损失分别计算，并通过权重组合为总损失。
匈牙利匹配用于计算预测点
'''
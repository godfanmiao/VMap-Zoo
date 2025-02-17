好的！如果要将解码模块也使用Transformer架构，我们需要进一步扩展模型，使其在特征融合后通过一个Transformer解码器进行预测。以下是完整的代码实现，其中编码和解码部分都使用Transformer架构。
修改后的代码
Python复制
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from scipy.optimize import linear_sum_assignment

# 定义模型
class MapTRModel(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
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
        self.output_fc = nn.Linear(embed_dim, 3)  # 输出类别ID和二维坐标

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
        predictions = self.output_fc(decoded_features.permute(1, 0, 2))  # [batch_size, seq_len, 3]
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
    def __init__(self, lambda_cls=1.0, lambda_pts=5.0, lambda_dir=1.0):
        super(MapTRLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_pts = lambda_pts
        self.lambda_dir = lambda_dir
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

        # 分类损失
        cls_pred = predictions[:, :, 0]
        cls_gt = targets[:, :, 0].long()
        cls_loss = self.classification_loss(cls_pred, cls_gt)

        # 匈牙利匹配
        cost_matrix = torch.zeros(batch_size, num_preds, num_gts, device=predictions.device)
        for i in range(batch_size):
            for j in range(num_preds):
                for k in range(num_gts):
                    cost_matrix[i, j, k] = torch.norm(predictions[i, j, 1:] - targets[i, k, 1:], p=2)
        matches = hungarian_matching(cost_matrix)

        # 点对点损失
        pts_loss = 0
        for i in range(batch_size):
            matched_gts = targets[i, matches[i]]
            pts_loss += self.point_loss(predictions[i, :, 1:], matched_gts[:, 1:])
        pts_loss /= batch_size

        # 方向损失（可选）
        # 这里简化为点对点损失，实际中可以根据需求添加方向损失的计算
        dir_loss = pts_loss

        # 总损失
        total_loss = self.lambda_cls * cls_loss + self.lambda_pts * pts_loss + self.lambda_dir * dir_loss
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

dataset = MapTRDataset(satellite_images, vector_points, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 初始化模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MapTRModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = MapTRLoss()

# 训练模型
train(model, dataloader, optimizer, criterion, device)

修改点说明
解码模块：
使用nn.TransformerDecoder作为解码模块，输入为初始化的零张量（tgt）和融合后的特征序列（fused_features）。
解码器通过自注意力和编码器-解码器注意力机制生成预测结果。
特征融合：
将卫星图像特征和矢量点图特征拼接后输入到Transformer编码器中进行融合。
融合后的特征作为解码器的输入。
输出模块：
解码器的输出通过一个全连接层（output_fc）生成最终的预测结果。
注意事项
Transformer解码器的输入需要初始化，这里使用零张量作为初始输入。
Transformer的性能对超参数（如embed_dim、num_heads、num_encoder_layers、num_decoder_layers）较为敏感，需要根据具体任务进行调整。
示例数据是随机生成的，实际使用时需要替换为真实数据，并进行适当的预处理。
希望这段代码能够满足你的需求！
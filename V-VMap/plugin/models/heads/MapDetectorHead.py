import copy
import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_positional_encoding
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn import bias_init_with_prob

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

@HEADS.register_module()
class MapDetectorHead(nn.Module):

    def __init__(self, num_classes=4,in_channels=256,embed_dims=256,num_queries = 100,num_points = 20):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_points = num_points

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=in_channels, drop_path=0.1) for j in range(2)]
            )
            self.stages.append(stage)

        self.m = nn.AdaptiveMaxPool2d((10, 10))

        self.norm_out = LayerNorm(self.embed_dims, eps=1e-6)

        cls_branch = nn.Linear(self.embed_dims, self.num_classes)

        reg_branch = [
            nn.Linear(self.embed_dims, 2 * self.embed_dims),
            nn.LayerNorm(2 * self.embed_dims),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dims, 2 * self.embed_dims),
            nn.LayerNorm(2 * self.embed_dims),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dims, self.num_points * 2),
        ]
        reg_branch = nn.Sequential(*reg_branch)

        cls_branches = nn.ModuleList(
            [copy.deepcopy(cls_branch) for _ in range(3)])
        reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(3)])

        self.reg_branches = reg_branches
        self.cls_branches = cls_branches

        self._init_embedding()
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # init prediction branch
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        bias_init = bias_init_with_prob(0.01)
        if isinstance(self.cls_branches, nn.ModuleList):
            for m in self.cls_branches:
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, bias_init)

    def _init_embedding(self):
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=self.embed_dims//2,
            normalize=True
        )
        self.bev_pos_embed = build_positional_encoding(positional_encoding)
        self.input_proj = nn.Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

    def _prepare_context(self, bev_features):
        """Prepare class label and vertex context."""
        # Add 2D coordinate grid embedding
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W)
        bev_pos_embeddings = self.bev_pos_embed(bev_mask)  # (bs, embed_dims, H, W)
        bev_features = self.input_proj(bev_features) + bev_pos_embeddings  # (bs, embed_dims, H, W)

        assert list(bev_features.shape) == [B, self.embed_dims, H, W]
        return bev_features

    def forward(self, bev_features):

        x = self._prepare_context(bev_features)

        outputs_class = []
        outputs_coord = []

        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            out = self.m(x)
            out = out.flatten(2).permute(0, 2, 1)
            out = self.norm_out(out)
            outputs_class.append(self.cls_branches[i](out))
            outputs_coord.append(self.reg_branches[i](out).sigmoid())

        return torch.stack(outputs_class), torch.stack(outputs_coord)

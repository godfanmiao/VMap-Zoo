import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import NECKS

class UpsampleBlock(nn.Module):
    def __init__(self, ins, outs):
        super(UpsampleBlock, self).__init__()
        self.gn = nn.GroupNorm(32, outs)
        self.conv = nn.Conv2d(ins, outs, kernel_size=3,
                              stride=1, padding=1)  # same
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(self.gn(x))
        x = self.upsample2x(x)

        return x

    def upsample2x(self, x):
        _, _, h, w = x.shape
        x = F.interpolate(x, size=(h*2, w*2),
                          mode='bilinear', align_corners=True)
        return x


class Upsample(nn.Module):

    def __init__(self,
                 zoom_size=(2, 4, 8),
                 in_channels=128,
                 out_channels=128,
                 ):
        super(Upsample, self).__init__()

        self.out_channels = out_channels

        input_conv = UpsampleBlock(in_channels, out_channels)
        inter_conv = UpsampleBlock(out_channels, out_channels)

        fscale = []
        for scale_factor in zoom_size:

            layer_num = int(math.log2(scale_factor))
            if layer_num < 1:
                fscale.append(nn.Identity())
                continue

            tmp = [copy.deepcopy(input_conv), ]
            tmp += [copy.deepcopy(inter_conv) for i in range(layer_num-1)]
            fscale.append(nn.Sequential(*tmp))

        self.fscale = nn.ModuleList(fscale)
        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):

        rescale_i = []
        for f, img in zip(self.fscale, imgs):
            rescale_i.append(f(img))

        out = sum(rescale_i)

        return out

@NECKS.register_module()
class FPN(nn.Module):
    def __init__(self,in_channels=[256, 256, 256],out_channels=256):
        super(FPN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.top_down_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn_top_down_conv1 = nn.BatchNorm2d(out_channels)

        self.top_down_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn_top_down_conv2 = nn.BatchNorm2d(out_channels)

        self.unsample = Upsample(zoom_size=(1, 2, 4),in_channels=out_channels,out_channels=out_channels)
        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        C0, C1, C2 = x

        P2 = self.bn1(self.conv1(C2))
        P1 = self.bn2(self.conv2(C1)) + F.interpolate(P2, scale_factor=2, mode='nearest')
        P0 = self.bn3(self.conv3(C0)) + F.interpolate(P1, scale_factor=2, mode='nearest')

        P0 = self.bn_top_down_conv2(self.top_down_conv2(P0))
        P1 = self.bn_top_down_conv1(self.top_down_conv1(P1))

        x = self.unsample([P0,P1,P2])

        return x

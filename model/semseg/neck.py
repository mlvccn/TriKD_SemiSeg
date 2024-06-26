from tkinter import W
import torch.nn as nn
from torch.nn import functional as F
import math


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride=1, padding=0, norm_layer=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class MultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.
    """
    def __init__(self, 
                 in_channels=[768, 768, 768, 768],
                 out_channels=768,
                 in_index=(2, 5, 8, 11),
                 scales=[0.5, 1, 2, 4]):
        super(MultiLevelNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.in_index = in_index
        
        norm_layer = nn.BatchNorm2d

        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                ConvBNReLU(in_channel, out_channels, 1, norm_layer=norm_layer)
            )
        for _ in range(self.num_outs):
            self.convs.append(
                ConvBNReLU(out_channels, out_channels, 3, padding=1, norm_layer=norm_layer)
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, inputs):
        # 选取部分输出
        feats = []
        for idx in self.in_index:
            feats.append(self.to_2D(inputs[idx]))
        inputs = feats
        assert len(inputs) == len(self.in_channels)
        inputs = [
            lateral_conv(inputs[i]) \
                for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # for len(inputs) not equal to self.num_outs
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            x_resize = F.interpolate(inputs[i], scale_factor=self.scales[i], mode='bilinear', align_corners=True, recompute_scale_factor=True)
            outs.append(self.convs[i](x_resize))
        return tuple(outs)
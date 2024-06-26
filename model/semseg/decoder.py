import torch
import torch.nn as nn
from torch.nn import functional as F
from model.semseg.fft_attn import FeatureSelectionModule_FFT



class PPM(nn.Module):
    def __init__(self, in_channels, channels, pool_scales, norm_layer=nn.BatchNorm2d):
        super(PPM, self).__init__()
        self.features = []
        for pool_scale in pool_scales:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


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



# UperHead_FSM_FFT2 base on UperHead_FSM_FFT, but will be given the output_channels for all stage
# params == 11:
#             in_channels = [64, 128, 256, 448]
#             num_heads=[2, 4, 8, 14]
#             out_channels = [48, 96, 192, 320]
class UperHead_FSM_FFT2(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=[256, 256, 256, 256], in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), dropout_ratio=0.1, num_classes=19, align_corners=False):
        super(UperHead_FSM_FFT2, self).__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index
        self.align_corners = align_corners

        norm_layer = nn.BatchNorm2d

        # PSP Module
        self.psp_module = PPM(self.in_channels[-1], out_channels[-1], pool_scales, norm_layer=norm_layer)
        self.psp_bottleneck = ConvBNReLU(self.in_channels[-1] + len(pool_scales) * out_channels[-1], out_channels[-1], 3, padding=1, norm_layer=norm_layer)
        

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        # 对laterals进行初始化，大致如下，对0，1层使用高斯高频滤波，对3层进行高斯低频滤波，对除了第一个阶段意外的阶段都要进行减少通道数的处理
        for stage, in_channel in enumerate(self.in_channels):
            # l_conv = ConvBNReLU(in_channel, self.channels, 1, norm_layer=norm_layer)
            # 如果不是最后一层
            # decoder消融改动
            if stage != len(self.in_channels) - 1:
                l_conv = FeatureSelectionModule_FFT(
                    in_channel, 
                    out_channels[stage], 
                    # low_flag = False if stage == 0 else True
                    # 要对第三层使用低通滤波器
                    low_flag = False if stage in [0, 1] else True
                )
                fpn_conv = ConvBNReLU(out_channels[stage], out_channels[stage], 3, padding=1, norm_layer=norm_layer)

                self.lateral_convs.append(l_conv)
                self.fpn_convs.append(fpn_conv)
            # 
            if stage != 0:
                down_conv = nn.Conv2d(out_channels[stage], out_channels[stage-1], kernel_size=1)
                self.down_convs.append(down_conv)

        
        self.fpn_bottleneck = ConvBNReLU(sum(out_channels), 256, 3, padding=1, norm_layer=norm_layer)

        self.cls_seg = nn.Sequential(
            # nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_ratio),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i]) \
                for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_bottleneck(self.psp_module(inputs[-1])))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] = laterals[i-1] + F.interpolate(
                self.down_convs[i-1](laterals[i]), # laterals[i] 改成了这一句
                size = prev_shape, 
                mode='bilinear', 
                align_corners=self.align_corners)
        
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) \
                for i in range(used_backbone_levels - 1)
        ]

        # append psp features
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)

        # Head
        output = self.cls_seg(feats)
        return output 


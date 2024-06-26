import importlib
import torch.nn as nn
from torch.nn import functional as F
import math
import model.backbone.resnet as resnet

from .decoder import UperHead_FSM_FFT2
from model.backbone.vit_kd import VisionTransformer
from model.semseg.neck import MultiLevelNeck
from model.backbone.tinyvit_kd import *

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


# clone from bisenet-stdc
class BiseNetHead(nn.Module):
    def __init__(self, in_channels, mid_channels, classes):
        super(BiseNetHead, self).__init__()
        self.conv3x3_bn_relu = ConvBNReLU(in_channels=in_channels, out_channels=mid_channels, ks=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels=mid_channels, out_channels=classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv3x3_bn_relu(x)
        out = self.conv1x1(x)
        return out


class ResUperBuilder(nn.Module):
    def __init__(self, cfg):
        super(ResUperBuilder, self).__init__()
        self._num_classes = cfg["num_classes"]

        model_name= cfg["encoder_cnn"]["type"]
        print("The model_name of CNN: ", model_name)

        self.encoder = resnet.__dict__[model_name](pretrained=True)

        if "resnet18" in model_name or "resnet34" in model_name:
            in_channels=[64, 128, 256, 512]
            channels=128
        else:
            in_channels=[256, 512, 1024, 2048]
            out_channels = [128, 256, 256, 512]
            print("The out_channels of CNN: ", out_channels)

        self.decoder = UperHead_FSM_FFT2(
            in_channels=in_channels, 
            out_channels=out_channels,
            in_index=[0, 1, 2, 3], 
            dropout_ratio=0.1, 
            num_classes=self._num_classes, 
            align_corners=True
        )
    
    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder.base_forward(x)
        outs = self.decoder(feats)
        outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
        if self.training:
            # feat_cnn_low = self.binary_head(feats[0])
            # feat_cnn_low = self.low_head(feats[0])
            # return outs, feat_cnn_low
            return outs, feats[0]
        return outs



class ViTUperBuilder(nn.Module):
    def __init__(self, net_cfg, img_size):
        super(ViTUperBuilder, self).__init__()
        self._num_classes = net_cfg["num_classes"]
        self.size = net_cfg["encoder_vit"]["type"]
        # assert self.size in ["small", "base", "large"]
        if self.size == "large":
            # encoder
            model_name='vit_large_patch16_384'
            embed_dim=1024
            depth=24
            num_heads=16
            mlp_ratio=4.
            qkv_bias=True
            drop_rate=0.1
            pos_embed_interp=True
            align_corners=False
            # neck
            in_channels=[1024, 1024, 1024, 1024]
            neck_channels=1024
            in_index=(5, 11, 17, 23)
        elif self.size == "base":
            # encoder
            model_name='vit_base_patch16_384'
            embed_dim=768
            depth=12
            num_heads=12
            mlp_ratio=4.
            qkv_bias=True
            drop_rate=0.0
            pos_embed_interp=True
            align_corners=True
            # neck
            in_channels=[768, 768, 768, 768]
            neck_channels=768
            in_index=(2, 5, 8, 11)
        elif self.size == "small":
            # encoder
            model_name='vit_small_patch16_224'
            embed_dim=768
            depth=8
            num_heads=8
            mlp_ratio=3.
            qkv_bias=False
            drop_rate=0.0
            pos_embed_interp=True
            align_corners=True
            # neck
            in_channels=[768, 768, 768, 768]
            neck_channels=768
            in_index=(1, 3, 5, 7)
        elif self.size == "deit_small":
            model_name='vit_deit_small_distilled_patch16_224'
            embed_dim=384
            depth=12
            num_heads=6
            mlp_ratio=4.
            qkv_bias=True
            drop_rate=0.0
            pos_embed_interp=True
            align_corners=True
            # neck
            in_channels=[384, 384, 384, 384]
            neck_channels=384
            in_index=(2, 5, 8, 11) # for depth=12
        elif self.size == "deit_base":
            # encoder
            model_name='vit_deit_base_distilled_patch16_384'
            embed_dim=768
            depth=12
            num_heads=12
            mlp_ratio=4.
            qkv_bias=True
            drop_rate=0.0
            pos_embed_interp=True
            align_corners=True
            # neck
            in_channels=[768, 768, 768, 768]
            neck_channels=448 # 768, 512, 448, 384, 256, 192, 128, 48
            in_index=(2, 5, 8, 11)
            out_channels = [48, 96, 192, 256]
            print("The out_channels of ViT: ", out_channels)
        print("The model_name of ViT: ", model_name)
        self.in_index = in_index
        print("The neck_channels of ViT: ", neck_channels)
        # self.low_head = nn.Conv2d(in_channels[0], 256, 1)

        self.encoder = VisionTransformer(
            model_name=model_name, 
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=self._num_classes,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            pos_embed_interp=pos_embed_interp,
            align_corners=align_corners
        )
        self.neck = MultiLevelNeck(
            in_channels=in_channels,
            out_channels=neck_channels,
            in_index=in_index,
            scales=[0.5, 1, 2, 4]
        )
        self.decoder = UperHead_FSM_FFT2(
            in_channels=[neck_channels]*4, 
            out_channels=out_channels,
            in_index=[0, 1, 2, 3], 
            dropout_ratio=0.1, 
            num_classes=self._num_classes, 
            align_corners=True
        )
    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x
    
    def forward(self, x):
        h, w = x.shape[-2:]
        feats, attns = self.encoder(x)
        feats_neck = self.neck(feats)
        outs = self.decoder(feats_neck)
        outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
        if self.training:
            # attn_stage3 = attns[self.in_index[-2]]
            attn_stage4 = attns[self.in_index[-1]]
            #只用于消融实验
            # feat_tiny_low = self.low_head(F.interpolate(self.to_2D(feats[0]),size=(128,128),mode="bilinear", align_corners=True))
            return outs, attn_stage4
        return outs
    
    


class TinyViTUperBUilder(nn.Module):
    def __init__(self, net_cfg, img_size) -> None:
        assert img_size in [224, 512, 768]
        super(TinyViTUperBUilder, self).__init__()
        self._num_classes = net_cfg["num_classes"]
        self.params = net_cfg["encoder_tiny"]["params"]
        assert self.params in [5, 11, 21]
        self.in_index = (2, 4, 6)
        model_name = f'tiny_vit_{self.params}m_{img_size}'
        print("The model_name of TinyViT: ", model_name)
        self.encoder = eval(model_name)(
            pretrained=True,
            num_classes=self._num_classes
        )

        if self.params == 5:
            in_channels = [64, 128, 160, 320]
        elif self.params == 11:
            in_channels = [64, 128, 256, 448]
            num_heads=[2, 4, 8, 14]
            out_channels = [48, 96, 192, 320]
            print("The out_channels of TinyViT: ", out_channels)

        else:
            in_channels = [96, 192, 384, 576]
            num_heads=[3, 6, 12, 18]
            out_channels = [48, 96, 192, 320]
            # out_channels = [96, 192, 384, 576]
            print("The out_channels of TinyViT: ", out_channels)
        # channels=512
        channels=256

        self.decoder = UperHead_FSM_FFT2(
            in_channels=in_channels, 
            out_channels=out_channels,
            in_index=[0, 1, 2, 3], 
            dropout_ratio=0.1, 
            num_classes=self._num_classes, 
            align_corners=True
        )
        # self.low_head = BiseNetHead(in_channels[0], 64, 1)
        self.low_head = nn.Conv2d(in_channels[0], 256, 1)
        vit_head = 12
        # self.high_head1 = nn.Conv2d(num_heads[-2], vit_head, 1)
        # self.high_head2 = nn.Conv2d(num_heads[-1], vit_head, 1)
        self.high_head = nn.Conv2d(num_heads[-1], vit_head, 1)

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x):
        h, w = x.shape[-2:]
        feats, attns = self.encoder(x)
       
        inputs = [feats[0]]
        for idx in self.in_index:
            inputs.append(self.to_2D(feats[idx]))
        outs = self.decoder(inputs)
        outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
        if self.training:
            feat_tiny_low = self.low_head(feats[0])
            # feat_tiny_low = feats[0]
            attn_stage4 = self.high_head(attns[-1][-1])
            # return outs, feats, attns
            return outs, feat_tiny_low, attn_stage4
        return outs
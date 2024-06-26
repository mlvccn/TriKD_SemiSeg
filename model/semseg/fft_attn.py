import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F
import numpy as np


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


class FeatureFrequencySeparationModule(nn.Module):
    """Feature Frequency Separation Module
    """
    def __init__(self, low_flag=True):
        super(FeatureFrequencySeparationModule, self).__init__()
        self.low_flag = low_flag
        print("self.low_flag: ", self.low_flag)

    def guassian_low_high_pass_filter(self, x, D0=10):
        """reference code: https://blog.csdn.net/weixin_43959755/article/details/115528425 
        """
        _, _, H, W = x.size()
        y, z = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        center = ((H-1)//2, (W-1)//2)
        dis_square = (y - center[0])**2 + (z - center[1])**2
        low_filter = torch.exp((-1) * dis_square / (2 * (D0 ** 2))).cuda()
        high_filter = 1 - low_filter
        return x * low_filter, x * high_filter
          
    def forward(self, x):
        # separate feature for two frequency
        fp16 = False
        # print(x.dtype)
        if str(x.dtype) == "torch.float16":
            fp16 = True
            x = x.float()
        # print(x.dtype)
        
        # freq_x = fft.fftn(x)
        freq_x = fft.fft2(x)
        freq_shift = fft.fftshift(freq_x)

        low_freq_shift, high_freq_shift = self.guassian_low_high_pass_filter(freq_shift, D0=10)

        if self.low_flag: 
            low_freq_ishift = fft.ifftshift(low_freq_shift)
            # _low_freq_x = torch.abs(fft.ifftn(low_freq_ishift))
            _low_freq_x = torch.abs(fft.ifft2(low_freq_ishift))
            if fp16 == True:
                _low_freq_x = _low_freq_x.half()
            # print("_low_freq_x.dtype: ", _low_freq_x.dtype)
            return _low_freq_x
        else:
            high_freq_ishift = fft.ifftshift(high_freq_shift)
            # _high_freq_x = torch.abs(fft.ifftn(high_freq_ishift))
            _high_freq_x = torch.abs(fft.ifft2(high_freq_ishift))
            if fp16 == True:
                _high_freq_x = _high_freq_x.half()
            # print("_high_freq_x.dtype: ", _high_freq_x.dtype)
            return _high_freq_x
        
        # low_freq_shift, high_freq_shift = self.guassian_low_high_pass_filter(freq_shift)

        # low_freq_ishift = fft.ifftshift(low_freq_shift)
        # high_freq_ishift = fft.ifftshift(high_freq_shift)
        
        # _low_freq_x = torch.abs(fft.ifft2(low_freq_ishift))
        # _high_freq_x = torch.abs(fft.ifft2(high_freq_ishift))
        # return _low_freq_x, _high_freq_x
        # low_freq_x = self.low_project(_low_freq_x)
        # high_freq_x = self.high_project(_high_freq_x)
        # return low_freq_x, high_freq_x



class FeatureSelectionModule_FFT(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN", low_flag=True):
        super(FeatureSelectionModule_FFT, self).__init__()
        self.fft_separate = FeatureFrequencySeparationModule(low_flag)
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.init_weight()
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
        
    def forward(self, x): # x: ([1, 64, 128, 128])
        freq_x = self.fft_separate(x) # freq_xx: ([1, 64, 128, 128])
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(freq_x, x.size()[2:]))) # F.avg_pool2d(freq_x, x.size()[2:]): ([1, 64, 1, 1])
        # attn: ([1, 64, 1, 1])
        feat = torch.mul(x, atten) # feat: ([1, 64, 128, 128])
        x = x + feat # x: ([1, 64, 128, 128])
        feat = self.conv(x)
        return feat


# base on FeatureSelectionModule_FFT, but without the last element-wise summation(x = x + feat)
class FeatureSelectionModule_FFT_wo_skip(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN", low_flag=True):
        super(FeatureSelectionModule_FFT_wo_skip, self).__init__()
        self.fft_separate = FeatureFrequencySeparationModule(low_flag)
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.init_weight()
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
        
    def forward(self, x): # x: ([1, 64, 128, 128])
        freq_x = self.fft_separate(x) # freq_xx: ([1, 64, 128, 128])
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(freq_x, x.size()[2:]))) # F.avg_pool2d(freq_x, x.size()[2:]): ([1, 64, 1, 1])
        # attn: ([1, 64, 1, 1])
        feat = torch.mul(x, atten) # feat: ([1, 64, 128, 128])
        # x = x + feat # x: ([1, 64, 128, 128])
        # feat = self.conv(x)
        feat = self.conv(feat)
        return feat

# base on FeatureSelectionModule_FFT, but add bn_relu for last conv1x1
class FeatureSelectionModule_FFT_BN_ReLU(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN", low_flag=True, norm_layer=nn.BatchNorm2d):
        super(FeatureSelectionModule_FFT_BN_ReLU, self).__init__()
        self.fft_separate = FeatureFrequencySeparationModule(low_flag)
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.conv = ConvBNReLU(in_chan, out_chan, ks=1, norm_layer=norm_layer)
        self.init_weight()
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
        
    def forward(self, x): # x: ([1, 64, 128, 128])
        freq_x = self.fft_separate(x) # freq_xx: ([1, 64, 128, 128])
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(freq_x, x.size()[2:]))) # F.avg_pool2d(freq_x, x.size()[2:]): ([1, 64, 1, 1])
        # attn: ([1, 64, 1, 1])
        feat = torch.mul(x, atten) # feat: ([1, 64, 128, 128])
        x = x + feat # x: ([1, 64, 128, 128])
        feat = self.conv(x)
        return feat


# https://github.com/moskomule/senet.pytorch
class SELayer_FFT(nn.Module):
    def __init__(self, channel, reduction=16, low_flag=True):
        super(SELayer_FFT, self).__init__()
        self.fft_separate = FeatureFrequencySeparationModule(low_flag)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        freq_x = self.fft_separate(x)
        y = self.avg_pool(freq_x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
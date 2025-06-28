import torch
import torch.nn as nn
from utils import model_info

class TransformerNetworkXiNet(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=4, lite=False, num_pool=0, bn_instead_of_in=False):
        super(TransformerNetworkXiNet, self).__init__()
        self.num_pool = num_pool
        norm = 'instancenorm2d' if not bn_instead_of_in else 'batchnorm'
        self.ConvBlock = nn.Sequential(
            XiConv(3, 16*alpha, 9  if not lite else 3, pool = 1 if not lite else 2, norm=norm, compression=gamma),
            XiConv(16*alpha, 64*alpha, 3, pool = 2, norm=norm, compression=gamma),
            XiConv(64*alpha, 128*alpha*beta, 3, pool = 2, norm=norm, compression=gamma),
        )
        self.ResidualBlock = nn.Sequential(
            XiResidual(int(128*alpha*beta), 3, norm='instance' if not bn_instead_of_in else 'batch'), 
            XiResidual(int(128*alpha*beta), 3, norm='instance' if not bn_instead_of_in else 'batch'), 
            XiResidual(int(128*alpha*beta), 3, norm='instance' if not bn_instead_of_in else 'batch'), 
            XiResidual(int(128*alpha*beta), 3, norm='instance' if not bn_instead_of_in else 'batch'), 
            XiResidual(int(128*alpha*beta), 3, norm='instance' if not bn_instead_of_in else 'batch')
        )
        self.DeconvBlock = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            XiConv(128*alpha*beta, 64*alpha, 3, norm=norm, compression=gamma),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            XiConv(64*alpha, 32*alpha, 3, norm=norm, compression=gamma),
            torch.nn.Upsample(scale_factor=2, mode='nearest') if lite else torch.nn.Identity(),
            XiConv(32*alpha, 16*alpha, 3, norm=norm, compression=gamma),
            ConvLayer(int(16*alpha), 3, 9 if not lite else 5, 1, norm="None"),
            nn.Sigmoid()
        )

        model_info(self, img_size=160)

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        out=out.mul(255.0)
        return out
    
    def forward_multiscale(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        out=out.mul(255.0)
        ret = [out]
        x = out
        for i in range(self.num_pool):
            x = nn.functional.avg_pool2d(x, 2)
            ret.append(x)
        return ret
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

class XiResidual(nn.Module):
    def __init__(self, channels=128, kernel_size=3, norm='instance'):
        super(XiResidual, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, norm=norm)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, norm=norm)

    def forward(self, x):
        identity = x                     
        out = self.relu(self.conv1(x))   
        out = self.conv2(out)            
        out = out + identity             
        return out


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class XiConv(nn.Module):
    # XiNET convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, compression=4, attention=False, skip_tensor_in=None, skip_channels=1, pool=None, upsampling=1, attention_k=3, attention_lite=True, norm='instancenorm2d', dropout_rate=0,  skip_k=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.compression = compression
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c2//compression
        self.pool = pool
        self.norm = norm
        self.dropout_rate = dropout_rate
        self.upsampling = upsampling

        c1=int(c1)
        c2=int(c2)

        self.compression_conv = nn.Conv2d(c1, c2//compression, 1, 1,  groups=g, padding=0, bias=False)
        self.main_conv = nn.Conv2d(c2//compression if compression>1 else c1, c2, k, s,  groups=g, padding=k//2 if s==1 else autopad(k, p), bias=False, padding_mode='reflect')
        self.act = nn.ReLU6() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
        if attention:
            if attention_lite:
                self.att_pw_conv= nn.Conv2d(c2, self.attention_lite_ch_in, 1, 1, groups=g, padding=0, bias=False)
            self.att_conv = nn.Conv2d(c2 if not attention_lite else self.attention_lite_ch_in, c2, attention_k, 1, groups=g, padding=attention_k//2, bias=False)
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool)
        if skip_tensor_in:
            self.skip_conv = nn.Conv2d(skip_channels, c2//compression, skip_k, 1,  groups=g, padding=skip_k//2, bias=False)

        if norm=='instancenorm2d':
            self.bn = nn.InstanceNorm2d(c2, affine=True)
        elif norm=='batchnorm':
            self.bn = nn.BatchNorm2d(c2, affine=True)

        if dropout_rate>0:
            self.do = nn.Dropout(dropout_rate)
        


    def forward(self, x):
        s = None
        # skip connection
        if isinstance(x, list):
            s = nn.functional.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            s = self.skip_conv(s)
            x = x[0]

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)
            
        if s is not None:
            x = x+s

        if self.pool:
            x = self.mp(x)
        if self.upsampling > 1:
            x = nn.functional.interpolate(x, scale_factor=self.upsampling, mode='nearest')
        # main conv and activation
        x = self.main_conv(x)
        if self.norm:
            x = self.bn(x)
        x = self.act(x)

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in=self.att_pw_conv(x)
            else:
                att_in=x
            y = self.att_act(self.att_conv(att_in))
            x = x*y

        
        if self.dropout_rate > 0:
            x = self.do(x)

        return x

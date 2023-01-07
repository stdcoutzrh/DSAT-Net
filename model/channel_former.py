import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from torch.nn import TransformerEncoderLayer

import math

def bchw2bcl(x):
    b,c,h,w = x.shape
    return x.view(b,c,h*w).contiguous()

def bcl2bchw(x):
    b,c,l = x.shape
    h = int(math.sqrt(l))
    w = h
    return x.view(b,c,h,w).contiguous()

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvChannelEmbeding(nn.Module):
    def __init__(self,in_c,down_ratio):
        super(ConvChannelEmbeding,self).__init__()

        self.in_c = in_c 
        self.channel_embeding = nn.Sequential(
            nn.Conv2d(in_c,in_c,down_ratio,down_ratio,0,groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.ReLU6()
        )

    def forward(self,x):
        x = self.channel_embeding(x)  
        return x        

class ChannelAttnBlock(nn.Module):
    def __init__(self,in_c,down_ratio,h,heads):
        """
        Channel Mixing Feature Refine Module
        """
        super(ChannelAttnBlock,self).__init__()

        self.in_c = in_c
        self.down_ratio = down_ratio
        self.dim = int((h//down_ratio)*(h//down_ratio))
        self.heads = heads

        self.ce = ConvChannelEmbeding(self.in_c,self.down_ratio)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.in_c, self.dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.attn = nn.Sequential(
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4),
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4),
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4),
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4)
        )

        self.up = nn.UpsamplingBilinear2d(scale_factor=self.down_ratio)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def forward(self,x):    
        x = self.ce(x) 
        shortcut = x
        x = bchw2bcl(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.attn(x)
        x = bcl2bchw(x) + shortcut
        x = self.up(x)
        return x       

    
if __name__ == "__main__":

    layer = ChannelAttnBlock(64,16,256,8)
    x = torch.ones([2,64,256,256])

    print(layer(x).shape)
 
    if 0:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(layer, x)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        if 1:
            total_paramters = 0
            for parameter in layer.parameters():
                i = len(parameter.size())
                p = 1
                for j in range(i):
                    p *= parameter.size(j)
                total_paramters += p
            print("Params: %.4f M" % (total_paramters / 1e6)) 
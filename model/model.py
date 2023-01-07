from pickle import NONE
import torch
import torch.nn as nn

import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .convnext import  convnext_encoder
from .spatial_former import SpatialFormer
from .channel_former import ChannelAttnBlock


class LRDU(nn.Module):
    """
    large receptive detailed upsample
    """
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in,2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Model(nn.Module):
    def __init__(self, n_class=2, pretrained = True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3

        self.Up5 = up_conv(ch_in=768, ch_out=384)
        self.Up_conv5 = conv_block(ch_in=384*2, ch_out=384)

        self.Up4 = up_conv(ch_in=384, ch_out=192)
        self.Up_conv4 = conv_block(ch_in=384, ch_out=192)

        self.Up3 = up_conv(ch_in=192, ch_out=96)
        self.Up_conv3 = conv_block(ch_in=192, ch_out=96)

        self.Up4x = LRDU(96,4)      
        self.convout = nn.Conv2d(96, n_class, kernel_size=1, stride=1, padding=0)

        self.decoder = True
        
        if 1:   # for 512x512
            self.ce1 = SpatialFormer(96,128)
            self.ce2 = SpatialFormer(192,64)
            self.ce3 = SpatialFormer(384,32)
            self.channel_mixer = ChannelAttnBlock(192,down_ratio=16,h=128,heads=2)
            
        self.backbone = convnext_encoder(pretrained,True)

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

    def forward(self, x):

        x128,x64,x32,x16 = self.backbone(x)

        """
        torch.Size([1, 96, 128, 128])
        torch.Size([1, 192, 64, 64])
        torch.Size([1, 384, 32, 32])
        torch.Size([1, 768, 16, 16])
        """

        d32 = self.Up5(x16) # [1, 384, 32, 32]
        d32 = torch.cat([x32,d32],dim=1) # [1, 786, 32, 32]
        d32 = self.Up_conv5(d32)   # 768->384
        if self.decoder:
            d32 =self.ce3(d32) + d32

        d64 = self.Up4(d32)  # 384-> 192
        d64 = torch.cat([x64,d64],dim=1)
        d64 = self.Up_conv4(d64)
        if self.decoder:
            d64 =self.ce2(d64) + d64

        d128 = self.Up3(d64) # [1,96,128,128]
        d128 = torch.cat([x128,d128],dim=1)
        d128 = self.channel_mixer(d128) + d128
        d128 = self.Up_conv3(d128)
        if self.decoder:
            d128 =self.ce1(d128) + d128 # [1,96,128,128]

        d2 = self.Up4x(d128)
        d1 = self.convout(d2)
        return d1

if __name__ == "__main__":

    model = Model(2,False)

    img = torch.rand((1,3,512,512))
    output = model(img)

    print(output.shape)
    
    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 

        #FLOPs: 57.7532 G
        #Params: 48.4985 M

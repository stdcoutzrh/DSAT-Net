import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math

class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''
    def __init__(self, in_c, init_value=1e-2):
        super().__init__()
        self.inChannels = in_c
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((in_c)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1)
            return scale * x

class DWConv3x3(nn.Module):
    def __init__(self, in_c):
        super(DWConv3x3, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True, groups=in_c)

    def forward(self, x):
        x = self.conv(x)
        return x

def bchw2bcl(x):
    b,c,h,w = x.shape
    return x.view(b,c,h*w).contiguous()

def bchw2blc(x):
    b,c,h,w = x.shape
    return x.view(b,c,h*w).permute(0,2,1).contiguous()

def bcl2bchw(x):
    b,c,l = x.shape
    h = int(math.sqrt(l))
    w = h
    return x.view(b,c,h,w).contiguous()

def blc2bchw(x):
    b,l,c = x.shape
    h = int(math.sqrt(l))
    w = h
    return x.view(b,h,w,c).permute(0,3,1,2).contiguous()

class FFN(nn.Module):
    def __init__(self, in_c, out_c, hid_c, ls=1e-2,drop=0.0):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_c)

        self.fc1 = nn.Conv2d(in_c, hid_c, 1)
        self.dwconv = DWConv3x3(hid_c)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_c, out_c, 1)

        self.layer_scale = LayerScale(in_c, init_value=ls)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        
        shortcut = x.clone()

        # ffn
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        # layer scale
        x = self.layer_scale(x)
        x = self.drop(x)

        out = shortcut + x
        return out


class LocalConvAttention(nn.Module):

    def __init__(self, dim):
        super(LocalConvAttention, self).__init__()
        
        # aggression local info
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) 

        self.conv0_1 = nn.Conv2d(dim, dim, (1,5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5,1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1) # channel mixer

    def forward(self, x):
        shortcut = x.clone()
        
        x_33 = self.conv0(x)

        x_15 = self.conv0_1(x)
        x_15 = self.conv0_2(x_15)

        x_111 = self.conv1_1(x)
        x_111 = self.conv1_2(x_111)

        add = x_33 + x_15 + x_111
        
        mixer = self.conv3(add)
        out = mixer * shortcut

        return out

class GlobalSelfAttentionV3(nn.Module):
    def __init__(self, dim, h ,drop=0.0):
        super(GlobalSelfAttentionV3, self).__init__()
        
        # aggression local info
        self.local_embed = nn.Sequential(
            nn.Conv2d(dim, dim//4, 4, 4 , groups=dim//4),
            nn.BatchNorm2d(dim//4),
            nn.ReLU6())

        self.dim = dim//4
        self.real_h = int(h//4)
        self.window_size = [self.real_h,self.real_h]
        
        if self.dim <=64:
            self.num_heads = 2
        else:
            self.num_heads = 4

        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(drop)

        self.conv_out = nn.Conv2d(self.dim, dim, 1, 1 )
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
                
        # local embed
        x = self.local_embed(x) # b c h w
        b,c,h,w = x.shape  
        x = x.view(b,c,h*w).permute(0,2,1).contiguous() # blc torch.Size([1, 256, 64])

        # self-attn
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        x = blc2bchw(x)
        out = self.up(self.conv_out(x))
       
        return out

class SpatialFormer(nn.Module):
    """
    DSAFormer
    """
    def __init__(self, dim, h, ls=1e-2, drop=0.0, vis = False):
        super(SpatialFormer,self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.detail_attn = LocalConvAttention(dim)
        self.global_attn = GlobalSelfAttentionV3(dim,h)

        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls)
        self.drop = nn.Dropout(p=drop)

        hidden_dim = 4*dim
        self.ffn = FFN(in_c=dim, out_c=dim, hid_c=hidden_dim, ls=ls, drop=drop)
        self.vis = vis
        
    def forward(self, x):

        shortcut = x.clone()

        # proj1
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        
        # attn    
        xd = self.detail_attn(x)
        xg = self.global_attn(x)
        attn = xd + xg

        # proj2
        attn = self.proj2(attn)
        attn = self.layer_scale(attn)
        attn = self.drop(attn)

        attn_out = attn + shortcut

        # ffn
        out = self.ffn(attn_out)

        if self.vis:
            fms = [shortcut,xd,xg,xd+xg,out]
            return out,fms
        else:
            return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math
import numpy as np

def MacPI2SAI(x, angRes):
  return rearrange(x, 'b c (h u) (w v) -> b c (u h) (v w)', u=angRes, v=angRes)


def SAI2MacPI(x, angRes):
  return rearrange(x, 'b c (u h) (v w) -> b c (h u) (w v)', u=angRes, v=angRes)


class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                # nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.Conv3d(inp, init_channels, (1, kernel_size, kernel_size), dilation=1,
                          padding=(0, kernel_size // 2, kernel_size // 2), bias=False),

                # nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                # nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.Conv3d(init_channels, new_channels, (1, dw_size, dw_size), dilation=1,
                          padding=(0, dw_size // 2, dw_size // 2), groups=init_channels, bias=False),
                # nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv3d(inp, init_channels, (1,kernel_size,kernel_size), dilation=1, padding=(0,kernel_size // 2,kernel_size//2), bias=False),
                # nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv3d(init_channels, new_channels, (1, dw_size, dw_size), dilation=1, padding=(0,dw_size//2, dw_size // 2), groups=init_channels, bias=False),
                # nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                # nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (stride, stride, stride), (0, kernel_size // 2, kernel_size//2), bias=False),
                # nn.BatchNorm2d(oup),
                # nn.Conv3d(oup, oup, kernel_size=(1, 1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.Conv3d(oup, oup, kernel_size=(1, 1, 5),  dilation=1, padding=(0, 0, 2), groups=oup, bias=False),

                # nn.BatchNorm2d(oup),
                # nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.Conv3d(oup, oup, kernel_size=(1, 5, 1),  dilation=1, padding=(0, 2, 0), groups=oup,
                                 bias=False)
                # nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            pool3d = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            res = self.short_conv(pool3d)
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-3], out.shape[-2], out.shape[-1]), mode='nearest')

class GhostModule2D(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModule2D, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]),
                                                           mode='nearest')

def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            pool3d = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            res = self.short_conv(pool3d)
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-3], out.shape[-2], out.shape[-1]), mode='nearest')


class get_model(nn.Module):
    def __init__(self,args):
        super(get_model, self).__init__()
        self.channels = args.channels
        self.angRes = args.angRes_in
        self.factor = args.scale_factor

        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 6
        self.MHSA_params['dropout'] = 0.

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            # nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            GhostModuleV2(self.channels, self.channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode="original"),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            GhostModuleV2(self.channels, self.channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,
                          mode="original"),
            nn.LeakyReLU(0.2, inplace=True),
            GhostModuleV2(self.channels, self.channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,
                          mode="original"),
            # nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = body(angRes=self.angRes)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, lr, info=None):
        # Bicubic
        lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')
        # [B(atch), 1, A(ngRes)*h(eight)*S(cale), A(ngRes)*w(idth)*S(cale)]

        # reshape for LFT
        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        # [B, C(hannels), A^2, h, w]

        # Initial Convolution
        buffer = self.conv_init0(lr)
        buffer = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]

        # Alternate AngTrans & SpaTrans
        buffer = self.altblock(buffer) + buffer

        # Up-Sampling
        buffer = rearrange(buffer, 'b c (a1 a2) h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        buffer = self.upsampling(buffer)
        out = buffer + lr_upscale

        return out


class AltFilter(nn.Module):
    def __init__(self, angRes):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.spa_trans = CSWinIR(is_spa=True, transtype=1, angRes=angRes)
        self.ang_trans = CSWinIR(is_spa=False, transtype=2, angRes=angRes)

        self.softmax_ = nn.Softmax(dim=2)
        # self.conv_de = nn.Conv3d(60*3, 60, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False)
        self.conv_de_1 = GhostModuleV2(60, 20, mode= "original")
        self.conv_de_2 = GhostModuleV2(60, 20, mode="original")
        self.conv_de_3 = GhostModuleV2(60, 20, mode="original")
        self.EPIFilter = EPIFilter(angRes=angRes)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, resAng, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # Reshape操作，将通道扩展为两维
        x = x.view(batchsize, groups, channels_per_group, resAng, height, width)
        # Transpose操作，将组卷积两个维度进行置换
        x = torch.transpose(x, 1, 2).contiguous()
        # Flatten操作，两个维度平展成一个维度
        x = x.view(batchsize, -1, resAng, height, width)

        x = x.view(batchsize, groups, channels_per_group, resAng, height, width)

        return x

    def forward(self, buffer):
        buffer1 = self.ang_trans(buffer)
        buffer2 = self.spa_trans(buffer)
        buffer3 = self.EPIFilter(buffer)
        buffer_tmp = torch.cat([buffer1,buffer2,buffer3],dim = 1)
        buffer_tmp = self.channel_shuffle(buffer_tmp, 3)


        buffer1 = self.conv_de_1(torch.squeeze(buffer_tmp[:, 0, :, :, :, :],1))
        buffer2 = self.conv_de_2(torch.squeeze(buffer_tmp[:, 1, :, :, :, :],1))
        buffer3 = self.conv_de_3(torch.squeeze(buffer_tmp[:, 2, :, :, :, :],1))

        buffer = torch.cat([buffer1,buffer2,buffer3],dim = 1) + buffer
        return buffer

class EPIFilter(nn.Module):
    def __init__(self, angRes):
        super(EPIFilter, self).__init__()

        self.epiw_trans = CSWinIR(is_spa=False, transtype=3, angRes=angRes)
        self.epih_trans = CSWinIR(is_spa=False, transtype=4, angRes=angRes)

    def forward(self, buffer):
        buffer = self.epiw_trans(buffer)
        buffer = self.epih_trans(buffer)

        return buffer


class body(nn.Module):
    def __init__(self, angRes):
        super(body, self).__init__()
        self.AltFilter1 = AltFilter(angRes=angRes)
        self.AltFilter2 = AltFilter(angRes=angRes)
        self.AltFilter3 = AltFilter(angRes=angRes)
        self.AltFilter4 = AltFilter(angRes=angRes)
        self.AltFilter5 = AltFilter(angRes=angRes)
        self.AltFilter6 = AltFilter(angRes=angRes)
        self.AltFilter7 = AltFilter(angRes=angRes)
        self.AltFilter8 = AltFilter(angRes=angRes)
        self.EPIFilter = EPIFilter(angRes=angRes)

    def forward(self, buffer):

        buffer1 = self.AltFilter1(buffer)
        buffer2 = self.AltFilter2(buffer1)
        buffer3 = self.AltFilter3(buffer2)
        buffer4 = self.AltFilter4(buffer3)
        buffer5 = self.AltFilter5(buffer4)
        buffer6 = self.AltFilter6(buffer5)
        buffer7 = self.AltFilter7(buffer6)
        buffer8 = self.AltFilter8(buffer7)

        buffer9 = self.EPIFilter(buffer8)
        return buffer9

def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor,
                                                                      W * scale_factor)  # [B, 1, A*h*S, A*w*S]
    return x_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, info=None):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):
    pass


class LeFF(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Sequential(nn.Linear(in_features, hidden_features),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, groups=hidden_features, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_features, out_features))
        self.dim = in_features
        self.hidden_dim = hidden_features
        # self.conv_1x3 = nn.Conv3d(in_features, in_features//2, kernel_size=(1, 1, 3), dilation = 1, padding=(0,0,1))
        # self.conv_1x3_2 = nn.Conv3d(in_features//2, in_features, kernel_size=(1, 1, 3), dilation = 1, padding=(0,0,1))
        # self.relu = nn.ReLU(inplace=True)
    def forward(self, x, x_size):
        # bs x hw x c
        B, L, C = x.shape
        H, W = x_size

        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)

        # x = x.view(L, C, 1, 1, B)
        # x = self.relu(self.conv_1x3_2(self.conv_1x3(x)))
        # x = x.view(B, L, C)
        return x

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:  # global attenton
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:  # row attention
            H_sp, W_sp = self.resolution[0], self.split_size
        elif idx == 1:  # column attention
            W_sp, H_sp = self.resolution[1], self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        if not self.H_sp == self.split_size:
            self.H_sp = H
        if not self.W_sp == self.split_size:
            self.W_sp = W
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, x_size, func):
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, x_size, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H, W = x_size
        B, L, C = q.shape

        q = self.im2cswin(q, x_size)
        k = self.im2cswin(k, x_size)
        v = self.im2cswin(v, x_size)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = LeFF(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](x_size, qkv[:, :, :, :C // 2])
            x2 = self.attns[1](x_size, qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class PA(nn.Module):
    def __init__(self, dim, x_size, kernel_size):
        super().__init__()
        if kernel_size == 3:
            self.pa_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, groups=dim)
        elif kernel_size == 5:
            self.pa_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=2, groups=dim)

        self.sigmoid = nn.Sigmoid()
        self.x_size = x_size
        self.dim = dim

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, with_pos=True,
                 is_spa=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.with_pos = with_pos

        if is_spa:
            self.pos = PA(embed_dim, patches_resolution, kernel_size=3)
        else:
            self.pos = PA(embed_dim, patches_resolution, kernel_size=5)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, split_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 img_size=224, patch_size=4, is_spa=True):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.is_spa = is_spa

        self.residual_group = nn.ModuleList([
            CSWinBlock(dim=dim, num_heads=num_heads, reso=input_resolution, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[i], norm_layer=norm_layer)
            for i in range(depth)])

        if is_spa:
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None, is_spa=is_spa)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        x1 = x
        x = self.patch_embed(x)
        for layer in self.residual_group:
            x = layer(x, x_size)
        if self.is_spa:
            x = self.conv(self.patch_unembed(x, x_size)) + x1
        else:
            x = self.patch_unembed(x, x_size)
        return x


class CSWinIR(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=64, patch_size=1, embed_dim=60, depths=2,
                 split_size=1,
                 num_heads=6, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_chk=False, patch_norm=True, is_spa=True,
                 angRes=5, transtype=1, ):
        super().__init__()

        self.split_size = split_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.is_spa = is_spa
        self.angRes = angRes
        self.transtype = transtype

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.use_chk = use_chk

        self.pos_drop = nn.Dropout(p=drop_rate)

        y = torch.linspace(0, drop_path_rate, depths)
        dpr = [x.item() for x in y]  # stochastic depth decay rule

        self.layer = RSTB(dim=embed_dim,
                          input_resolution=(patches_resolution[0],
                                            patches_resolution[1]),
                          depth=depths,
                          num_heads=num_heads,
                          split_size=split_size,
                          mlp_ratio=self.mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=dpr,  # no impact on SR results
                          norm_layer=norm_layer,
                          img_size=img_size,
                          patch_size=patch_size,
                          is_spa=self.is_spa
                          )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def check_image_size(self, x, transtype):
        b, c, a, h, w = x.size()

        if transtype == 1:
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * a, c, h, w)
            mod_pad_h = (self.split_size - h % self.split_size) % self.split_size
            mod_pad_w = (self.split_size - w % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        elif transtype == 2:
            x = rearrange(x, 'b c (a1 a2) h w -> (b h w) c a1 a2', a1=self.angRes, a2=self.angRes)
            mod_pad_a = (self.split_size - self.angRes % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_a, 0, mod_pad_a), 'reflect')
        elif transtype == 3:
            x = rearrange(x, 'b c (a1 a2) h w -> (b h a1) c w a2', a1=self.angRes, a2=self.angRes)
            mod_pad_a = (self.split_size - self.angRes % self.split_size) % self.split_size
            mod_pad_w = (self.split_size - w % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_a, 0, mod_pad_w), 'reflect')
        elif transtype == 4:
            x = rearrange(x, 'b c (a1 a2) h w -> (b w a2) c h a1', a1=self.angRes, a2=self.angRes)
            mod_pad_a = (self.split_size - self.angRes % self.split_size) % self.split_size
            mod_pad_h = (self.split_size - h % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_a, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.pos_drop(x)
        x = self.layer(x, x_size)
        return x

    def forward(self, x):
        b, c, a, H, W = x.shape
        x = self.check_image_size(x, self.transtype)
        x = self.forward_features(x)
        if self.transtype == 2:
            x = rearrange(x, '(b h w) c a1 a2->(b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=H, w=W)
        if self.transtype == 3:
            x = rearrange(x, '(b h a1) c w a2->(b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=H, w=W)
        if self.transtype == 4:
            x = rearrange(x, '(b w a2) c h a1->(b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=H, w=W)
        x = x[:, :, :H, :W]
        x = x.view(b, a, c, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x


if __name__ == "__main__":
    from option import args
    net = get_model(args).cuda()
    from thop import profile

    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))

#    Number of parameters: 1.57M
#    Number of FLOPs: 39.05G

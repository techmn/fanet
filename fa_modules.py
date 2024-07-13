import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel//2, groups=in_dim)
        self.proj = nn.Conv2d(in_dim*2, out_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B,C,H,W = x.shape

        dx = self.down(x)
        udx = F.interpolate(dx, size=(H,W), mode='bilinear', align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))

        out = self.act(self.proj(torch.cat([lx, hx], dim=1))) 

        return out


class AFE(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim//2, 1, padding=0)
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0)

        self.ctx_conv = nn.Conv2d(dim//2, dim//2, kernel_size=7, padding=3, groups=4)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim//2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dim//2, eps=1e-6, data_format="channels_first")

        self.enhance = FeatureRefinementModule(in_dim=dim//2, out_dim=dim//2, down_kernel=3, down_stride=2)        
        
        self.act = nn.GELU()


    def forward(self, x):
        B, C, H, W = x.shape
        
        x = x + self.norm1(self.act(self.dwconv(x)))
        x = self.norm2(self.act(self.proj1(x)))
        
        ctx = self.norm3(self.act(self.ctx_conv(x)))

        enh_x = self.enhance(x)
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))

        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.1, expan_ratio=4,
                 kernel_size=3, use_dilated_mlp=False):
        super().__init__()
        
        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        if use_dilated_mlp:
            self.mlp = AtrousMLP(dim=dim, mlp_ratio=expan_ratio)
        else:
            self.mlp = MLP(dim=dim, mlp_ratio=expan_ratio)
        self.attn = AFE(dim, kernel_size=kernel_size)

        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        
    def forward(self, x):
        B, C, H, W = x.shape

        inp_copy = x
        x = self.layer_norm1(inp_copy)
        x = self.drop_path_1(self.attn(x))
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.drop_path_2(self.mlp(x))
        out = out + x

        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, use_dcn=False):
        super().__init__()
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class AtrousMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos1 = nn.Conv2d(dim * mlp_ratio, dim*2, 3, padding=1, groups=dim*2)
        self.pos2 = nn.Conv2d(dim * mlp_ratio, dim*2, 3, padding=2, dilation=2, groups=dim*2)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.act(self.fc1(x))
        x1 = self.act(self.pos1(x))
        x2 = self.act(self.pos2(x))
        x_a = torch.cat([x1,x2], dim=1)
        x = self.fc2(x_a)

        return x

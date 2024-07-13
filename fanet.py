import torch
from torch import nn
from timm.models.layers import trunc_normal_
from fa_modules import Block, LayerNorm
from mmseg.models.builder import BACKBONES


class FANet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, img_size=224,
                 depths=[2, 2, 8, 2], dims=[96, 96*2, 96*4, 96*8],
                 drop_path_rate=0.1, expan_ratio=4,
                 kernel_sizes=[5, 5, 3, 3], **kwargs):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        
        stem = nn.Conv2d(in_chans, dims[0], kernel_size=5, stride=4, padding=2)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                                          expan_ratio=expan_ratio, kernel_size=kernel_sizes[i], use_dilated_mlp=False)
                                    )

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

        #self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            checkpoint = torch.load(pretrained, map_location="cpu")
            msg = self.load_state_dict(checkpoint["state_dict"], strict=False)
            print(msg)

    def forward_features(self, x):
        feats = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        feats.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)

        return feats

    def forward(self, x):
        f1, f2, f3, f4 = self.forward_features(x)
        
        return [f1, f2, f3, f4]


@BACKBONES.register_module()
class fanet_tiny(FANet):
    def __init__(self, in_chans=3, depths=[2, 2, 8, 2], dims=[96, 96*2, 96*4, 96*8],
                 drop_path_rate=0.1, expan_ratio=4, kernel_sizes=[5, 5, 3, 3], **kwargs):
        super(fanet_tiny, self).__init__()
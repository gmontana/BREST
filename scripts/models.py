import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)  # Reduced number of intermediate channels

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature1, feature2):
        combined_features = feature1 + feature2

        local_att = self.local_att(combined_features)
        global_att = self.global_att(combined_features)

        combined_att = local_att + global_att
        att_weights = self.sigmoid(combined_att)

        fused_features = 2 * feature1 * att_weights + 2 * feature2 * (1 - att_weights)

        return fused_features

class AFF_LnR(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF_LnR, self).__init__()
        inter_channels = int(channels // r)  # Reduced number of intermediate channels

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature1, feature2):
        combined_features = feature1 + feature2

        local_att = self.local_att(combined_features)
        global_att = self.global_att(combined_features)

        combined_att = local_att + global_att
        att_weights = self.sigmoid(combined_att)

        weighted_feature1 = 2 * feature1 * att_weights
        weighted_feature2 = 2 * feature2 * (1 - att_weights)
        
        fused_features = torch.cat((weighted_feature1, weighted_feature2), dim=1)

        return fused_features

class FusionBlock(nn.Module):
    """
    4096-ch concat → 2048-ch fused map (28×28) with:
    - 1×1 squeeze (channel mixing)
    - depthwise-separable dilated 3×3 (spatial mixing)
    - CBAM attention
    - residual skip with matching 1×1 on the skip-path
    - GroupNorm throughout
    """
    def __init__(self, in_c=4096, out_c=2048, groups_dw=32):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

        self.dw_sep = nn.Sequential(
            # depthwise 3×3 (dilation 2 ➜ 7×7 effective RF)
            nn.Conv2d(out_c, out_c, 3, padding=2, dilation=2,
                      groups=groups_dw, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            # pointwise 1×1 to re-mix channels
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

        self.cbam    = CBAM(out_c)
        self.dropout = nn.Dropout2d(0.1)

        # skip-path to match channels
        self.skip_conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.skip_gn   = nn.BatchNorm2d(out_c)

    def forward(self, x):
        identity = self.skip_gn(self.skip_conv(x))          # [B, 2048, 28, 28]

        out = self.squeeze(x)                               # 1×1
        out = self.dw_sep(out)                              # depthwise + pw
        out = self.cbam(out)                                # attention
        out = self.dropout(out)

        return F.relu(out + identity)                       # residual + ReLU


# ───────────────────────── 1.  Bottleneck with optional dilation ─────────────────────────
class Bottleneck(nn.Module):
    """
    ResNeXt/ResNet bottleneck that supports group convs, stride, and dilation.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels,
                 stride: int = 1, dilation: int = 1, groups: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # 1×1 reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # 3×3 group conv  (stride + dilation as requested)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               dilation=dilation,
                               groups=groups,
                               bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 1×1 expansion
        self.conv3 = nn.Conv2d(out_channels,
                               out_channels * self.expansion,
                               1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        # skip-connection pathway
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


# ───────────────────────── 2.  make_layer now forwards a dilation argument ────────────────
def make_layer(block, in_channels, out_channels,
               num_blocks: int, stride: int = 1,
               dilation: int = 1, groups: int = 32):
    """
    Builds one ResNeXt stage (a sequence of Bottleneck blocks).
    The *first* block can change stride; all blocks share dilation.
    """
    layers = []
    layers.append(block(in_channels, out_channels,
                        stride=stride, dilation=dilation, groups=groups))
    in_channels = out_channels * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(in_channels, out_channels,
                            stride=1, dilation=dilation, groups=groups))
    return nn.Sequential(*layers)


# ───────────────────────── 3.  CBAM   ─────────────────────────────────────
class ChannelAttention(nn.Module):
    """Channel gate as in Woo et al., ECCV 2018."""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared   = nn.Sequential(
            nn.Conv2d(in_planes,  in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.shared(self.avg_pool(x)) +
                            self.shared(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv     = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(k)

    def forward(self, x):
        x = self.ca(x) * x
        return self.sa(x) * x

    
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # ResNeXt part
        self.resnext = models.resnext50_32x4d()
        
        # Remove the last FC layer of ResNeXt-101_32x8d(50_32x4d)
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-2])

        # Additional ResNeXt layers
        self.addLayers = nn.Sequential(
            # CBAM attention to ResNeXt-50 outputs
            CBAM(2048), nn.Dropout2d(0.1),
            
            # Stage-A  (dilation 1, stride 1)  → 56×56
            make_layer(Bottleneck, 2048, 512, 3, stride=1, dilation=1),
            CBAM(2048), nn.Dropout2d(0.1),

            # Stage-B  (dilation 2, stride 1)  → 56×56 (larger RF)
            make_layer(Bottleneck, 2048, 512, 3, stride=1, dilation=2),
            CBAM(2048), nn.Dropout2d(0.1),

            # Stage-C  (dilation 1, stride 2)  → 28×28
            make_layer(Bottleneck, 2048, 512, 3, stride=2, dilation=1),
            CBAM(2048), nn.Dropout2d(0.1),
        )
        
        # Fusion modules
        self.aff_CC_MLO = AFF(2048,4)
        self.aff_LnR    = AFF_LnR(2048,2)
        self.fusion     = FusionBlock(4096, 2048)
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1)
        )
        
    def forward(self, x):
        B, V, C, H, W = x.shape          # V=4 views
        x = x.view(-1, C, H, W)          # flatten view dimension
        
        # Pass input through ResNeXt
        features = self.resnext(x)
        
        # Pass through additional layers
        features = self.addLayers(features)
        
        _, C2, H2, W2 = features.shape
        features = features.view(B, V, C2, H2, W2)
        
        CC_L, CC_R, MLO_L, MLO_R = (features[:, 0], features[:, 1], features[:, 2], features[:, 3])
        
        features_L = self.aff_CC_MLO(CC_L, MLO_L)
        features_R = self.aff_CC_MLO(CC_R, MLO_R)
        features_Ep = self.aff_LnR(features_L, features_R)
        features_fused = self.fusion(features_Ep)
        
        pooled = F.adaptive_avg_pool2d(features_fused, 1).flatten(1)  # [B, 2048]

        # Pass features through MLP
        x = self.mlp(pooled)

        return x

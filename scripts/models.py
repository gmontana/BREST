import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4  # This is the expansion factor for ResNeXt, it's 4 for ResNet

    def __init__(self, in_channels, out_channels, stride=1, groups=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
def make_layer(block, in_channels, out_channels, num_blocks, stride=1, groups=32):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        layers.append(block(in_channels, out_channels, stride, groups))
        in_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

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
  
class CADModel(nn.Module):
    def __init__(self):
        super(CADModel, self).__init__()

        # ResNeXt part
        self.resnext = models.resnext50_32x4d()
        
        # Remove the last FC layer of ResNeXt-101_32x8d(50_32x4d)
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-2])

        # Additional ResNeXt layers
        self.addLayers = nn.Sequential(
        # Additional ResNeXt layers
            make_layer(Bottleneck, 2048, 1024, 6, stride=2),
            nn.Dropout2d(0.1),    
            make_layer(Bottleneck, 4096, 2048, 3, stride=2),
            nn.Dropout2d(0.1)
        )

        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Pass input through ResNeXt
        features = self.resnext(x)
        
        # Pass through additional layers
        features = self.addLayers(features)
        
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(x.size(0), -1)
        
        # Reshape and concatenate features
        features = features.view(batch_size, 8192)

        # Pass features through MLP
        x = self.mlp(features)

        return x
      
class RiskModel(nn.Module):
    def __init__(self):
        super(RiskModel, self).__init__()

        # ResNeXt part
        self.resnext = models.resnext50_32x4d()
        
        # Remove the last FC layer of ResNeXt-101_32x8d(50_32x4d)
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-2])

        # Additional ResNeXt layers
        self.addLayers = nn.Sequential(
        # Additional ResNeXt layers
            make_layer(Bottleneck, 2048, 1024, 6, stride=2),
            nn.Dropout2d(0.1),    
            make_layer(Bottleneck, 4096, 2048, 3, stride=2),
            nn.Dropout2d(0.1)
        )

        # MLP part
        self.MLP = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Pass input through ResNeXt
        features = self.resnext(x)
        
        # Pass through additional layers
        features = self.addLayers(features)
        
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(x.size(0), -1)
        
        # Reshape and concatenate features
        features = features.view(batch_size, 8192)

        # Pass features through MLP
        x = self.MLP(features)

        return x

class BRESTModel(nn.Module):
    def __init__(self):
        super(BRESTModel, self).__init__()

        # ResNeXt part
        self.resnext = models.resnext50_32x4d()
        
        # Remove the last FC layer of ResNeXt-101_32x8d(50_32x4d)
        self.resnext = nn.Sequential(
            *list(self.resnext.children())[:-2]
        )
        
        # Additional ResNeXt layers
        self.addLayers = nn.Sequential(
            make_layer(Bottleneck, 2048, 1024, 6, stride=2),
            nn.Dropout2d(0.1),    
            make_layer(Bottleneck, 4096, 2048, 3, stride=2),
            nn.Dropout2d(0.1)
        )
        
        self.aff_CC_MLO = AFF(channels=8192, r=4)
        self.aff_LnR = AFF_LnR(channels=8192, r=2)
        
        self.feature_fusion = nn.Sequential(
            make_layer(Bottleneck, 8192*2, 2048, 3, stride=1),
            nn.Dropout2d(0.1)
        )
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1)
        )
        

    def forward(self, x):
        batch_size, num_images, channels, height, width = x.size()
        x = x.view(batch_size * num_images, 3, 1792, 1792)  # Reshape to feed into ResNet
        
        # Pass input through ResNeXt
        features = self.resnext(x)
        
        # Pass through additional layers
        features = self.addLayers(features)
        
        f_batch_size, f_channels, f_height, f_width = features.size()
        
        features = features.view(batch_size, num_images, f_channels, f_height, f_width)
        
        # Reshape to [batch_size, height * width, channels]
        featuresCCL = features[:, 0, :, :, :].view(batch_size, f_channels, f_height, f_width)
        featuresCCR = features[:, 1, :, :, :].view(batch_size, f_channels, f_height, f_width)
        featuresMLOL = features[:, 2, :, :, :].view(batch_size, f_channels, f_height, f_width)
        featuresMLOR = features[:, 3, :, :, :].view(batch_size, f_channels, f_height, f_width)
        
        features_L = self.aff_CC_MLO(featuresCCL, featuresMLOL)
        features_R = self.aff_CC_MLO(featuresCCR, featuresMLOR)
        features_ALL = self.aff_LnR(features_L, features_R)
        
        features_ALL = self.feature_fusion(features_ALL)

        features_ALL = nn.functional.adaptive_avg_pool2d(features_ALL, (1, 1)).view(x.size(0), -1)
        features_ALL = features_ALL.view(batch_size, 8192)
        
        # Pass features through MLP
        x = self.mlp(features_ALL)

        return x

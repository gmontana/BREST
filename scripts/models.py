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
  
class CADModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

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
      
class CADModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

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

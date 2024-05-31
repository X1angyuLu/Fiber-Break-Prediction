import torch
import itertools
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from dataloader import dataset
from matplotlib import pyplot as plt

class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(4 * growth_rate)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)  # Concatenate along the channel axis
        return out

class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList([Bottleneck3D(in_channels + i * growth_rate, growth_rate)
                                     for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool3d(out, kernel_size=2, stride=2)
        return out

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16)):
        super(DenseNet3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        in_channels = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(in_channels, growth_rate, num_layers)
            self.features.add_module('denseblock%d' % (i + 1), block)
            in_channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition3D(in_channels, in_channels // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                in_channels = in_channels // 2

        self.features.add_module('norm5', nn.BatchNorm3d(in_channels))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        # self.features.add_module('avgpool', nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.linear1 = nn.Linear(8192, 8192)
        self.linear2 = nn.Linear(8192, 4048)

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = out.view(out.size(0), -1)  
        out = self.linear1(out)
        out = self.linear2(out)
        return out
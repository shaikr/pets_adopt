# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch.nn.functional as F
from torch import nn

from layers.conv_layer import conv3x3


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class RegNet(nn.Module):

    def __init__(self):
        super(RegNet, self).__init__()
        self.in_planes = [3, 20, 40, 60, 80]
        kernel_sizes = [11, 9, 7, 5]
        # input size is 750x500
        self.conv1 = nn.Conv2d(self.in_planes[0], self.in_planes[1], kernel_size=kernel_sizes[0], stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.in_planes[1])

        # input size is: (740x490) / 2 = 370x245
        self.conv2 = nn.Conv2d(self.in_planes[1], self.in_planes[2], kernel_size=kernel_sizes[1], stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.in_planes[2])

        # input size is: (362x237) / 2 = 181x118
        self.conv3 = nn.Conv2d(self.in_planes[2], self.in_planes[3], kernel_size=kernel_sizes[2], stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.in_planes[3])

        # input size is: (175 x 112) / 2 = 87 x 56
        self.conv4 = nn.Conv2d(self.in_planes[3], self.in_planes[4], kernel_size=kernel_sizes[3], stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(self.in_planes[4])

        # input size is: (83 x 52) / 2 = 41 x 26
        self.linear1 = nn.Linear(1066, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x):

        out = F.max_pool2d(self.bn1(F.relu(self.conv1(x))), kernel_size=2, stride=2)
        out = F.max_pool2d(self.bn2(F.relu(self.conv2(out))), kernel_size=2, stride=2)
        out = F.max_pool2d(self.bn3(F.relu(self.conv3(out))), kernel_size=2, stride=2)
        out = F.max_pool2d(self.bn4(F.relu(self.conv4(out))), kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out

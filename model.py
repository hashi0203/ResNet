import torch.nn as nn
import torch.nn.functional as F

# ボトルネックブロックの作成
class Bottleneck(nn.Module):
    expansion = 4

    def __init__ (self, ich, mch, stride=1):
        super(Bottleneck, self).__init__()
        och = mch * self.expansion

        # 1 x 1
        self.conv1 = nn.Conv2d(ich, mch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mch)

        # 3 x 3
        self.conv2 = nn.Conv2d(mch, mch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mch)

        # 1 x 1
        self.conv3 = nn.Conv2d(mch, och, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(och)

        self.shortcut = nn.Sequential()
        if stride != 1 or ich != och:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ich, och, 1, stride=stride, bias=False),
                nn.BatchNorm2d(och)
            )

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = self.bn3(h)
        h += self.shortcut(x)
        h = F.relu(h)
        return h

# ResNet の作成
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.ich = 64

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.bottleneck1 = self._make_layer(64, num_blocks[0])
        self.bottleneck2 = self._make_layer(128, num_blocks[1], stride=2)
        self.bottleneck3 = self._make_layer(256, num_blocks[2], stride=2)
        self.bottleneck4 = self._make_layer(512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512*Bottleneck.expansion, num_classes)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.bottleneck1(h)
        h = self.bottleneck2(h)
        h = self.bottleneck3(h)
        h = self.bottleneck4(h)

        h = F.avg_pool2d(h, h.size()[2:])
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

    def _make_layer(self, mch, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.ich, mch, stride))
            self.ich = mch * Bottleneck.expansion
        return nn.Sequential(*layers)

def ResNet101(num_classes):
    return ResNet([3, 4, 23, 3], num_classes)

def ResNet152(num_classes):
    return ResNet([3, 8, 36, 3], num_classes)
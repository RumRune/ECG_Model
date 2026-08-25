"""
一维 ResNet 模型（用于 ECG 多标签分类）
已加入 Squeeze-and-Excitation 模块，并移除最后的 Sigmoid（输出 logits）
"""
import torch
import torch.nn as nn


def conv3x1(in_planes, out_planes, stride=1):
    """3x1 卷积（扩大时间感受野）"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer1D(nn.Module):
    """Squeeze-and-Excitation 块（一维版本）"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.2):
        super().__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer1D(planes, reduction=16)          # 新增 SE 模块

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)                     # 注意力加权

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """一维 ResNet，支持多标签分类（输出 logits，无 Sigmoid）"""

    def __init__(self, block, layers, in_channels=12, num_classes=10, dropout=0.2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], dropout=dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # ❌ 删除 self.sigmoid = nn.Sigmoid()

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dropout=0.2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 12, 1000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)          # (batch, 512, 1)
        x = torch.flatten(x, 1)      # (batch, 512)
        x = self.fc(x)               # (batch, 10)  ← 直接返回 logits
        return x


def resnet18_1d(in_channels=12, num_classes=10, dropout=0.2):
    """ResNet-18 一维版本"""
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], in_channels, num_classes, dropout)


def resnet34_1d(in_channels=12, num_classes=10, dropout=0.2):
    """ResNet-34 一维版本"""
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], in_channels, num_classes, dropout)


if __name__ == "__main__":
    # 快速测试模型结构
    model = resnet18_1d()
    x = torch.randn(2, 12, 1000)
    y = model(x)
    print(f"输出尺寸: {y.shape}")  # (2, 10)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
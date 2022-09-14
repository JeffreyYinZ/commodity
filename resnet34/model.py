import torch.nn as nn
import torch


# 算法各层具体参数可以参考项目目录下 ResNet结构.jpg
class BasicBlock(nn.Module):  # 创建基本残差单元
    expansion = 1  # 扩张倍率

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):  # 输入通道，输出通道，卷积步长，是否下采样
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)  # 第一个卷积层（输入通道，输出通道，卷积核尺寸，卷积步长，padding尺寸，是否偏置）
        self.bn1 = nn.BatchNorm2d(out_channel)  # 算法权重和偏置项归一化
        self.relu = nn.ReLU()  # Relu激活函数
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1,
                               bias=False)  # 第二个卷积层（输入通道，输出通道，卷积核尺寸，卷积步长，padding尺寸，是否偏置）
        self.bn2 = nn.BatchNorm2d(out_channel)  # 算法权重和偏置项归一化
        self.downsample = downsample  # 是否执行下采样操作

    def forward(self, x):  # 前向传播函数，x为输入的数据
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 经过两次卷积后的数据和输入数据求和
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 算法基本单元
                 blocks_num,  # 基本单元重复次数
                 num_classes=1000,  # 样本类别数量
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups  # 分组数量为1，采用普通卷积
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)  # 算法第一个卷积层（输入通道，输出通道，卷积核尺寸，卷积步长，padding尺寸，是否偏置）

        self.bn1 = nn.BatchNorm2d(self.in_channel)  # 批量归一化
        self.relu = nn.ReLU(inplace=True)  # relu激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大值池化操作，尺寸为3x3， 步长为2， 填充一个像素值
        self.layer1 = self._make_layer(block, 64, blocks_num[0])             # 创建第一个残差组， 数量为3
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # 创建第二个残差组， 数量为4
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # 创建第三个残差组， 数量为6
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 创建第四个残差组， 数量为3
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均值池化操作，输出尺度为(1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，输出维度为num_classes，即分类数量

        for m in self.modules():  # 对各模块儿的卷积成权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # 构建残差组
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:  # 如果步长为2 则对其降维处理（作为每个残差组第一个残差模块的 合并项之一）
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []  # 创建空列表，用于存放各网络层
        layers.append(block(self.in_channel,  # 残差组的 第一个残差单元（输入通道，输出通道，降维操作，步长，卷积分组数量）
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion  # 下一个残差单元的输入通道（对于resnet34，expansion=1）

        for _ in range(1, block_num):  # 遍历残差组数量-1，在layer中添加剩余的残差单元
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 最大值池化

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)  # 全局平均值池化操作
            x = torch.flatten(x, 1)  # 将全局平局值池化后的网络层展平成向量
            x = self.fc(x)  # 全连接层完成分类

        return x


def resnet34(num_classes=1000, include_top=True):  # 定义Resnt34函数，调用时只需要传入分类数量参数即可
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

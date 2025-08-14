"""VGG11/13/16/19 在 PyTorch 中的实现

该模块对官方示例进行了轻量化修改，并加入中文注释以便理解。
在本地运行前请确保已经安装好 ``torch`` 等依赖库，
若需要在 GPU 上训练请提前配置好 CUDA 环境。
"""
# 修改自 https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch.nn as nn
from torch.autograd import Variable


# ``cfg`` 字典定义了不同 VGG 版本对应的网络结构。
# 数字表示卷积层输出通道数，'M' 表示 MaxPooling 层。
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """简单实现的 VGG 网络

    参数
    ------
    vgg_name: str
        指定使用哪一种 VGG 结构，如 ``'VGG11'``。
    num_classes: int
        最终分类的类别数量。
    """

    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        # 特征提取部分，由多个卷积层和池化层堆叠组成
        self.features = self._make_layers(cfg[vgg_name])
        # 分类器，用于输出各类别的预测分数
        self.classifier = nn.Linear(1024, num_classes)
        # 额外的置信度预测分支
        self.confidence = nn.Linear(1024, 1)

    def forward(self, x):
        """前向传播，返回分类结果和置信度"""
        out = self.features(x)
        out = out.view(out.size(0), -1)  # 展平为二维张量

        pred = self.classifier(out)
        confidence = self.confidence(out)

        return pred, confidence

    def _make_layers(self, cfg):
        """根据配置 ``cfg`` 构建卷积层序列"""
        layers = []
        in_channels = 2  # 输入为 2 个通道的 CSI 图像
        for x in cfg:
            if x == 'M':
                # 'M' 代表最大池化层，用于降低特征尺寸
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 卷积 -> BN -> ReLU 组合
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # 最后通过平均池化获得固定长度的特征向量
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

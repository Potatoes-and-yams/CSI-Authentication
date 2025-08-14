# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:30:41 2021

该文件实现了一个简单的卷积神经网络，用于 CSI 数据的身份识别。
在本地运行本模块前请先安装 ``torch`` 库，并准备好 GPU 或 CPU 环境。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """用于 CSI 认证的基础 CNN 模型"""

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 第一层卷积，将输入的 2 通道 CSI 图像映射到 20 通道
        self.conv1 = nn.Conv2d(2, 20, kernel_size=(5, 5), stride=(2, 2))
        # 第二层卷积，进一步提取特征
        self.conv2 = nn.Conv2d(20, 40, kernel_size=(5, 5), stride=(2, 2))
        self.conv2_drop = nn.Dropout2d()  # 随机失活部分通道，防止过拟合
        # 全连接层，将卷积得到的特征映射到隐藏空间
        self.fc1 = nn.Linear(240, 80)
        # 预测类别和置信度的输出层
        self.pred = nn.Linear(80, num_classes)
        self.confidence = nn.Linear(80, 1)

    def forward(self, x):
        """前向传播，输出分类结果和置信度"""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 240)  # 展平成一维向量
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        pred = self.pred(x)
        confidence = self.confidence(x)
        return pred, confidence


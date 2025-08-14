"""辅助工具函数集合

包含 One-Hot 编码、CSV 日志记录及 Cutout 数据增强等常用功能。
如需在本地环境中使用，请确保已安装 ``torch``、``numpy`` 等依赖。
"""

import torch
import csv      # 逗号分隔值格式，用于读写训练日志
import numpy as np


def encode_onehot(labels, n_classes):
    """将整数标签转为 one-hot 向量"""
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    # scatter_ 在指定维度根据索引填充 1，其余保持为 0
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


class CSVLogger():
    """简单的 CSV 日志记录器"""

    def __init__(self, args, filename='log.csv', fieldnames=['epoch']):
        self.filename = filename
        self.csv_file = open(filename, 'w')

        # 将实验配置写在文件头部，便于复现
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()  # 立即写入磁盘

    def close(self):
        self.csv_file.close()


class Cutout(object):
    """Randomly mask out one or more patches from an image.
       https://arxiv.org/abs/1708.04552
    Args:
        length (int): The length (in pixels) of each square patch.补丁
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if np.random.choice([0, 1]):
            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img

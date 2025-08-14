"""用于生成噪声数据集的工具

该模块提供高斯噪声与均匀噪声两种数据集实现，
可用于评估模型的鲁棒性。在本地使用时无需额外数据文件，
直接实例化相应类即可获得随机样本。
"""

import numpy as np
from torch.utils.data import Dataset


class GaussianNoise(Dataset):
    """高斯噪声数据集"""

    def __init__(self, size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0):
        self.size = size
        self.n_samples = n_samples
        self.mean = mean
        self.variance = variance
        # 生成符合高斯分布的随机数据
        self.data = np.random.normal(loc=self.mean, scale=self.variance, size=(self.n_samples,) + self.size)
        self.data = np.clip(self.data, 0, 1)  # 将数据限制在 [0,1] 区间
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


class UniformNoise(Dataset):
    """均匀噪声数据集"""

    def __init__(self, size=(3, 32, 32), n_samples=10000, low=0, high=1):
        self.size = size
        self.n_samples = n_samples
        self.low = low
        self.high = high
        # 生成均匀分布的随机数据
        self.data = np.random.uniform(low=self.low, high=self.high, size=(self.n_samples,) + self.size)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


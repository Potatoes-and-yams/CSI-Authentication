"""CSI 测试数据加载器

用于在模型验证或测试阶段读取 CSI 数据。
使用前请在 ``data`` 目录下准备好对应的 ``.mat`` 文件，
并安装依赖库 ``h5py``、``torch`` 等。
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py


class TestCSIDataLoad(Dataset):
    """CSI 测试数据集包装类"""

    def __init__(self, data_CSI, data_label, transform):
        self.data = data_CSI
        self.label = data_label
        self.transform = transform

    def __getitem__(self, index):
        """返回指定索引处的样本及标签"""
        data = self.data[:, :, :, index]
        data = self.transform(data)
        labels = self.label[index]
        return data, labels

    def __len__(self):
        """返回数据集中样本的数量"""
        return np.size(self.data, 3)


"""示例
-------
>>> source_data = torch.rand(10, 20, 10, 100)
>>> source_label = np.random.randint(0, 2, (10, 1))
>>> dataset = TestCSIDataLoad(source_data, source_label, transform)
"""


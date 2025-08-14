# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:56:38 2021
@author: wangshaoyu

该模块提供训练阶段使用的 CSI 数据加载器 ``TrainCSIDataLoad``。
使用前请在本地 ``data`` 目录中准备好 ``channel_ind.mat`` 等文件，
并确保已经安装依赖库 ``numpy``、``torch``、``h5py`` 等。
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import h5py
# 定义 CSIDataLoad 类，继承 ``Dataset`` 方法，并重写 ``__getitem__`` 和 ``__len__`` 方法
# =============================================================================
# class TrainCSIDataLoad(Dataset):
# 	# 初始化函数，得到数据
#     def __init__(self, data_CSI, data_label,transform1,transform2,transform3):
#         self.data = data_CSI
#         self.label = data_label
#         self.transform1 = transform1
#         self.transform2 = transform2
#         self.transform3 = transform3
#     # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
#     def __getitem__(self, index):
#         data = self.data[:,:,:,index]
#         #Image.fromarray(data)
#         if 0 <= index < 10000:
#             data = self.transform1(data)
#         elif 10000 <= index < 15000:
#             permutation1 = np.random.permutation(data.shape[0])
#             data = data[permutation1,:,:]
#             data = self.transform1(data)
#         elif 15000 <= index < 20000:
#             permutation2 = np.random.permutation(data.shape[1])
#             data = data[:,permutation2,:]
#             data = self.transform1(data)
# # =============================================================================
# #         elif 20000 <= index < 21000:
# #             permutation3 = np.random.permutation(data.shape[0])
# #             permutation4 = np.random.permutation(data.shape[1])
# #             data = data[permutation3,:,:]
# #             data = data[:,permutation4,:]
# #             data = self.transform2(data)
# # =============================================================================
#         elif 20000 <= index < 30000:
#             data = self.transform2(data)
#         else:
#             data = self.transform3(data)
#         labels = self.label[index]
#         return data, labels
#     # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，CSIDataLoad会一脸懵逼
#     def __len__(self):
#         #return len(self.data)
#         return np.size((self.data),3)
# =============================================================================

class TrainCSIDataLoad(Dataset):
    """CSI 训练数据加载器

    本类在 ``__getitem__`` 中根据样本索引选择不同的数据增强方式，
    用于提升模型的泛化能力。
    """

    def __init__(self, data_CSI, data_label, transform1, transform2, transform3, transform4):
        self.data = data_CSI
        self.label = data_label
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4

    def __getitem__(self, index):
        """根据索引返回样本及其标签，并施加相应的数据增强"""
        data = self.data[:, :, :, index]
        if 14000 <= index < 17000:
            permutation1 = np.random.permutation(data.shape[0])  # 交换行
            data = data[permutation1, :, :]
            data = self.transform1(data)
        elif 17000 <= index < 20000:
            data = self.transform2(data)  # 随机裁剪
        elif 20000 <= index < 23000:
            permutation2 = np.random.permutation(data.shape[1])  # 交换列
            data = data[:, permutation2, :]
            data = self.transform1(data)
        elif 23000 <= index < 30000:
            permutation3 = np.random.permutation(data.shape[0])  # 同时交换行列
            permutation4 = np.random.permutation(data.shape[1])
            data = data[permutation3, :, :]
            data = data[:, permutation4, :]
            data = self.transform1(data)
        elif 30000 <= index < 33000:
            data = self.transform3(data)  # Cutout 数据增强
        elif 33000 <= index < 40000:
            data = self.transform4(data)  # Cutout + 随机裁剪
        else:
            data = self.transform1(data)
        labels = self.label[index]
        return data, labels

    def __len__(self):
        """返回样本总量，便于 ``DataLoader`` 正确切分批次"""
        return np.size((self.data), 3)




        
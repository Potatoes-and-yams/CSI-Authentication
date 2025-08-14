# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:57:50 2021

@author: wangshaoyu
"""

# =============================================================================
# 本脚本用于在 CSI 数据集上训练认证模型。
# 本地运行前请完成以下准备工作：
# 1. 安装所需依赖库，例如运行 `pip install -r requirements.txt`。
# 2. 将采集得到的 `channel_ind.mat` 数据文件放入工程根目录下的 `data/` 文件夹。
# 3. 若 `logs/` 与 `checkpoints/` 目录不存在，请在项目根目录中手动创建：
#      `mkdir logs checkpoints`
# 4. 如需使用 GPU，请提前配置好 CUDA 环境。
# 5. 运行示例：`python main.py --dataset CSI --model vgg`
# =============================================================================
import numpy as np
import h5py               #读取.mat文件
import argparse
from tqdm import tqdm     #Python进度条库
from PIL import Image     #图像处理库
import skimage
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import  transforms
import seaborn as sns  # import this after torch or it will break everything
from test_CSI_data_load import TestCSIDataLoad
from train_CSI_data_load import TrainCSIDataLoad
from models.vgg import VGG
from models.cnn import CNN
from utils.utils import encode_onehot, CSVLogger, Cutout


parser = argparse.ArgumentParser(description='Authenticator')
parser.add_argument('--dataset', default='CSI')
parser.add_argument('--model', default='vgg')
parser.add_argument('--batch_size', type=int, default=180)
parser.add_argument('--epochs', type=int, default=36)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1) 
parser.add_argument('--budget', type=float, default=1, metavar='N',
                    help='the budget for how often the network can get hints')
args = parser.parse_args()


cudnn.benchmark = True  # Should make training should go faster for large models，选择合适的卷积算法
filename = '2022mainB_' + str(args.budget) + '_seed_' + str(args.seed)
np.random.seed(0)  #设置随机数，注意：仅一次有效，为np.random下的随机数生成函数设置种子
torch.cuda.manual_seed(args.seed) #为当前GPU设置随机数种子，使得每次实验的网络初始化参数一致，使得每次实验之间具有可比性

# CSI Data Preprocessing 对数据进行加噪，有利于信心分数的训练
#训练集处理
train_transform1 = transforms.Compose([transforms.ToTensor()])
train_transform2 = transforms.Compose([transforms.ToTensor(),transforms.RandomCrop((64,52), padding=6)])
train_transform3 = transforms.Compose([transforms.ToTensor(),Cutout(30)])
train_transform4 = transforms.Compose([transforms.ToTensor(),Cutout(30),transforms.RandomCrop((64,52), padding=4)])
#测试集不做处理，只做ToTensor即可
test_transform = transforms.Compose([transforms.ToTensor()])

#读取CSI数据
CSIDataFileLoad = h5py.File('data\\channel_ind.mat','r')  # 读取mat文件，请确保 data 目录下存在 channel_ind.mat
CSIData = np.transpose(CSIDataFileLoad['channel_ind'][:])#转置后才能和channel的维度一致
CSIData = CSIData.astype(np.float32)
CSIData = np.array(CSIData)


#生成CSI数据标签
CSILable = np.zeros(50000)
for i in range(0,10):
    CSILable[0+5000*i:5000+5000*i] = i*np.ones(5000)
CSILable = torch.from_numpy(CSILable)
CSILable = CSILable.long()    #onehot.scatter函数的要求 

 
#训练集和测试集划分
permutation = np.random.permutation(CSILable.shape[0])    
shuffled_CSIData = CSIData[:,:,:,permutation]
shuffled_CSILable = CSILable[permutation]

#添加高斯噪声，和交换行列共同作用，生成难以区分的样本
gaosi = np.random.normal(loc=0.0, scale=0.1, size=(64,52,2,14000))
junyun = np.random.uniform(low=-0.1, high=0.1, size=(64,52,2,14000))
shuffled_CSIData[:,:,:,:14000] = shuffled_CSIData[:,:,:,:14000] + gaosi + junyun
shuffled_CSIData = shuffled_CSIData.astype(np.float32)

train_CSIData = shuffled_CSIData[:,:,:,:40000]
test_CSIData = shuffled_CSIData[:,:,:,40000:]
train_CSILable = shuffled_CSILable[:40000]
test_CSILable = shuffled_CSILable[40000:]
train_dataset = TrainCSIDataLoad(train_CSIData, train_CSILable, train_transform1,train_transform2,train_transform3,train_transform4)
test_dataset = TestCSIDataLoad(test_CSIData, test_CSILable, test_transform)



num_classes = 10

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=0)



def test(loader):
    """在验证/测试集上评估模型表现"""
    cnn.eval()    # 切换为评估模式，BN 层使用滑动平均的均值和方差

    correct = []
    probability = []
    confidence = []

    for images, labels in loader:
        images = Variable(images).cuda()
        labels = labels.cuda()

        pred, conf = cnn(images)              # 前向推理得到预测类别和置信度
        pred = F.softmax(pred, dim=-1)        # 将原始输出转为概率分布
        conf = F.sigmoid(conf).data.view(-1)  # 置信度经过 Sigmoid 压缩到 0~1

        pred_value, pred = torch.max(pred.data, 1)   # 返回类别概率值和类别索引
        correct.extend((pred == labels).cpu().numpy())
        probability.extend(pred_value.cpu().numpy())
        confidence.extend(conf.cpu().numpy())

    correct = np.array(correct).astype(bool)
    probability = np.array(probability)
    confidence = np.array(confidence)

    val_acc = np.mean(correct)    # 准确率
    conf_min = np.min(confidence) # 置信度最小值
    conf_max = np.max(confidence) # 置信度最大值
    conf_avg = np.mean(confidence) # 置信度平均值

    cnn.train()  # 重新切换回训练模式
    return val_acc, conf_min, conf_max, conf_avg, confidence

if args.model == 'vgg':   #选择网络模型 
    cnn = VGG(vgg_name='VGG11', num_classes=num_classes).cuda()
elif args.model == 'cnn':
    cnn = CNN(num_classes=num_classes).cuda()
cnn = cnn.float()       #输入数据类型和网络参数类型一致

prediction_criterion = nn.NLLLoss().cuda()

#优化器设置
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=2e-4)

#scheduler = MultiStepLR(cnn_optimizer, milestones=[3,6,9,12,15,18,21,24,27], gamma=0.2)
scheduler = MultiStepLR(cnn_optimizer, milestones=[6,12,18,24,30,36], gamma=0.2)



csv_logger = CSVLogger(args=args, filename='logs/' + filename + '.csv',
                        fieldnames=['epoch', 'train_acc', 'test_acc'])  # 日志将保存在 logs 目录，若不存在请先创建

# ---------------------------- 模型训练循环 ----------------------------
# Start with a reasonable guess for lambda
lmbda = 0.1
for epoch in range(args.epochs):  # 遍历所有训练轮次

    xentropy_loss_avg = 0.
    confidence_loss_avg = 0.
    correct_count = 0.
    total = 0.
    progress_bar = tqdm(train_loader)  # 使用 tqdm 展示训练进度
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))  # 设置进度条前缀为当前 epoch

        images = Variable(images).cuda(non_blocking=True)
        labels = Variable(labels).cuda(non_blocking=True)
        labels_onehot = Variable(encode_onehot(labels, num_classes))  # 将标签转换为 one-hot 形式
        labels_onehot = labels_onehot.float()        # 输入数据类型和网络参数类型一致
        cnn.zero_grad() #梯度清零

        pred_original, confidence = cnn(images)  # 前向传播得到原始预测与置信度

        pred_original = F.softmax(pred_original, dim=-1)  # 一个样本的概率输出值是一行
        confidence = F.sigmoid(confidence)                # 置信度归一化到 0~1 范围
        
        # Make sure we don't have any numerical instability   确保没有数值波动
        eps = 1e-12
        pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)  #夹紧到区间
        confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

       
        # Randomly set half of the confidences to 1 (i.e. no hints) 一半数据没有提示，避免过度正则化
        b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()  # 随机掩码，用于随机去除提示
        conf = confidence * b + (1 - b)
        pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
        #*为按元素相乘，.expand_as为复制扩展，公式（2）得来的
        pred_new = torch.log(pred_new)
        

        xentropy_loss = prediction_criterion(pred_new, labels)  #通过softmax、log、NLLLoss之后，一个batch的交叉损失熵
        confidence_loss = torch.mean(-torch.log(confidence))    #一个batch的信心分数损失值

      
        total_loss = xentropy_loss + (lmbda * confidence_loss)  # 总损失 = 分类损失 + λ*置信度损失
        if args.budget > confidence_loss.data:  # 动态调整 λ 以满足置信度预算
            lmbda = lmbda / 1.01
        elif args.budget <= confidence_loss.data:
            lmbda = lmbda / 0.99
                      
        total_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.data
        confidence_loss_avg += confidence_loss.data
        pred_idx = torch.max(pred_original.data, 1)[1] #对softmax函数的输出值进行操作，求出预测值索引。函数会返回两个tensor，第一个tensor是每行的最大值，
        #softmax的输出中最大的是1，所以第一个tensor是全1的tensor；第二个tensor是每行最大值的索引。
        #total += labels.size(0)
        #correct_count += (pred_idx == labels.data).sum()
        
        total = labels.size(0)               # 当前批次样本数量
        correct_count = (pred_idx == labels.data).sum()  # 预测正确的样本数
        #print(correct_count.data)
        accuracy = correct_count.data / total
        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            confidence_loss='%.3f' % (confidence_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc, conf_min, conf_max, conf_avg, Aconfidence = test(test_loader)
    tqdm.write('test_acc: %.3f, conf_min: %.3f, conf_max: %.3f, conf_avg: %.3f' % (test_acc, conf_min, conf_max, conf_avg))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

    torch.save(cnn.state_dict(), 'checkpoints/' + filename + '.pt')  # 保存模型参数到 checkpoints 目录，本地运行前需确保该目录已创建
csv_logger.close()



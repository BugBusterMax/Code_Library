#coding : utf-8
#_author:"kid_wang"
#Date   :2021/5/10 13:44

#coding : utf-8
#_author:"kid_wang"
#Date   :2021/1/7 22:18

import torch
from torch_geometric.nn import TopKPooling, GCNConv, TAGConv,SAGPooling
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
import time
import random
from torch_sparse import spspmm
import math
# from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch_geometric.utils.repeat import repeat
from torch_geometric.utils import (add_self_loops, sort_edge_index,remove_self_loops)
import sys
sys.path.append('F:/Pycharm/Projects/New_idea/pygcn-master')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda',         action='store_true',  default=False,   help='Disables CUDA training.')
parser.add_argument('--fastmode',        action='store_true',  default=False,   help='Validate during training pass.')
parser.add_argument('--EPOCH',           type=int,             default=100,     help='Number of epochs to train.')
parser.add_argument('--LR',              type=float,           default=0.001,  help='Initial learning rate.')
parser.add_argument('--weight_decay',    type=float,           default=5e-4,    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--validation_step', type=int,             default=1,       help='check model for every X time')
parser.add_argument('--hidden',          type=int,             default=256,     help='Number of hidden units.')
parser.add_argument('--dropout',         type=float,           default=0.5,     help='Dropout rate (1 - keep probability).')
parser.add_argument('--edge',            type=float,           default=6,       help='numbers of sides.')
parser.add_argument('--mu',              type=float,           default=0,       help='Gauss noise mean.')
parser.add_argument('--sigma',           type=float,           default=5,       help='Gauss noise std.')
args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# data preprocess

df0 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\00春古.csv')
df1 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\01春越.csv')
df2 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\02古江.csv')
df3 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\03古舜.csv')
df4 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\04萧越.csv')
df5 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\05萧古.csv')

df01 = df0.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df11 = df1.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df21 = df2.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df31 = df3.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df41 = df4.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df51 = df5.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))

df0 = pd.concat([df01,df0.iloc[:,8:10]],axis=1)
df1 = pd.concat([df11,df1.iloc[:,8:10]],axis=1)
df2 = pd.concat([df21,df2.iloc[:,8:10]],axis=1)
df3 = pd.concat([df31,df3.iloc[:,8:10]],axis=1)
df4 = pd.concat([df41,df4.iloc[:,8:10]],axis=1)
df5 = pd.concat([df51,df5.iloc[:,8:10]],axis=1)
# print(df1.shape)                # [1635,9]

df0 = np.array(df0)     #将pandas数据结构转化为numpy格式
df1 = np.array(df1)
df2 = np.array(df2)
df3 = np.array(df3)
df4 = np.array(df4)
df5 = np.array(df5)


features = [[[[]for p in range(7)]for i in range(6)]for j in range(8640)]
labels1 = [[[[]for p in range(1)]for i in range(6)]for j in range(8640)]
labels2 = [[[[]for p in range(1)]for i in range(6)]for j in range(8640)]


#生成特征和标签
for row in range(0, df0.shape[0]):
    for j in range(0, 7):
        features[row][0][j] = df0[row][:][j]
        features[row][1][j] = df1[row][:][j]
        features[row][2][j] = df2[row][:][j]
        features[row][3][j] = df3[row][:][j]
        features[row][4][j] = df4[row][:][j]
        features[row][5][j] = df5[row][:][j]
#features. shape=(1635,6,7)

for row in range(0, df0.shape[0]):
    for j in range(1):
        labels1[row][0][j] = df0[row][:][j+7]
        labels1[row][1][j] = df1[row][:][j+7]
        labels1[row][2][j] = df2[row][:][j+7]
        labels1[row][3][j] = df3[row][:][j+7]
        labels1[row][4][j] = df4[row][:][j+7]
        labels1[row][5][j] = df5[row][:][j+7]
# labels1.shape=(1635, 6, 1)
for row in range(0, df0.shape[0]):
    for j in range(1):
        labels2[row][0][j] = df0[row][:][j+8]
        labels2[row][1][j] = df1[row][:][j+8]
        labels2[row][2][j] = df2[row][:][j+8]
        labels2[row][3][j] = df3[row][:][j+8]
        labels2[row][4][j] = df4[row][:][j+8]
        labels2[row][5][j] = df5[row][:][j+8]
# labels2.shape=(1635, 6, 1)
features = np.array(features, dtype=np.float32)
features = np.nan_to_num(features)
labels1 = np.array(labels1, dtype=np.float32)
# labels1 = labels1 * (10**7)
labels2 = np.array(labels2, dtype=np.float32)

# 加高斯噪声  SNR 50dB
features = np.swapaxes(features, 0,1)
labels1 = np.swapaxes(labels1, 0,1)

def gen_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)                         # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1/signal.shape[0]) * np.sum(np.power(signal,2))
    noise_variance = signal_power/np.power(10, (SNR/10))
    noise = (np.sqrt(noise_variance) / np.std(noise))*noise
    return noise

for i in range(args.edge):      #edge=6
    noise = gen_gaussian_noise(features[i], 50)
    features[i] = features[i] + noise

# for i in range(args.edge):
# #     noise = gen_gaussian_noise(labels1[i], 50)
# #     labels1[i] = labels1[i] + noise

features = np.swapaxes(features, 0,1)
labels1 = np.swapaxes(labels1, 0,1)
#
# Drop Edge
# N = 1 drop edge
for i in range(8640):
    for j in range(6):
        ssp = random.random()
        ssr = float(1/6)
        if ssp < ssr:
            features[i][j] = 0

# Drop Data
# N = 0.01 drop data
for i in range(8640):
    for j in range(6):
        for p in range(7):
            sd = random.random()
            sf = 0.01
            if sd < sf:
                features[i][j][p] = 0

features = torch.FloatTensor(features)
labels1 = torch.FloatTensor(labels1)
labels2 = torch.FloatTensor(labels2)

labels = torch.cat((labels1, labels2), dim=1)       #标签1，2按列合并
#分配训练集和测试集
idx_train = features[0:7776, :]
idx_test = features[7776:8640, :]

labels_train = labels[0:7776, :]
labels_test = labels[7776:8640, :]

def normalize(mx):
    #对邻接矩阵进行正则化
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def MAE(SR, GT):
    SR = SR.flatten()
    GT = GT.flatten()
    corr = torch.sum(abs(SR - GT))
    tensor_size = SR.size(0)
    mae = float(corr) / float(tensor_size)
    return mae

def MSE(SR, GT):
    SR = SR.flatten()
    GT = GT.flatten()
    corr = torch.sum((SR - GT)**2)
    tensor_size = SR.size(0)
    mse = float(corr) / float(tensor_size)
    return mse

def R(SR, GT):
    corr = MSE(SR, GT)
    varr = torch.var(GT)
    r = 1 - float(corr)/float(varr)
    return r

edge_index = torch.tensor([[0, 0, 0, 0, 0,1, 1, 1, 1, 1,2, 2, 2, 2, 2,3, 3, 3, 3, 3,4,4, 4, 4, 4,5, 5, 5, 5, 5],
                           [1, 2, 3, 4, 5,0, 2, 3, 4, 5,0, 1, 3, 4, 5,0, 1, 2, 4, 5,0, 1, 2, 3, 5,0, 1, 2, 3, 4]],dtype=torch.long)
edge_index = torch.LongTensor(edge_index).cuda()
#以下定义awl
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        # print('params', self.params)
        # print('params2', self.params[1])

        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            # loss_sum += 0.5 * torch.exp(-log_vars[i]) * loss + self.params[i]
        return loss_sum

def criterion(y_pred, y_true, log_vars):
    loss = 0
    for i in range(len(y_pred)):
        precision = torch.exp(-log_vars[i])
        diff = (y_pred[i]-y_true[i])**2. ## mse loss function
        loss += torch.sum(precision * diff + log_vars[i], -1)
    return torch.mean(loss)
#from UGPOOL import UGPool
#以下为定义u型块
from torch_geometric.nn import SuperGATConv
class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):            #sum_res：一个布尔标志，指示是否在网络中使用残差连接。如果设置为True，则每个层的输出会加上前一级网络中相应层的输出。
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()     #下采样开始
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(TAGConv(in_channels, channels,K=4))
        for i in range(depth):  #depth=2
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(TAGConv(channels, channels,K=4))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()       #上采样开始
        for i in range(depth - 1):
            self.up_convs.append(TAGConv(in_channels, channels,K=4))
        self.up_convs.append(TAGConv(in_channels, out_channels,K=4))


        self.reset_parameters()

    def reset_parameters(self):     #调用reset_parameters来初始化模型参数
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

#定义网络结构，一个u型结构接三个线性层
class MTUNet(torch.nn.Module):
    """
    backbone: graph U-Net
    """
    def __init__(self, nfeat, nhid, nclass, dep, dropout, pool_ratios, sum_res):
        super(MTUNet, self).__init__()
        self.graphu = GraphUNet(in_channels=nfeat, hidden_channels=nhid, out_channels=nclass, depth=dep, pool_ratios=0.8, sum_res=True)
        # self.graphu1 = TAGConv(in_channels=nfeat, out_channels=nhid, K=2)
        # self.graphu2 = TAGConv(in_channels=nhid, out_channels=nclass, K=2)

        self.liner1_1 = nn.Linear(16 * args.edge, 256)         # 256 128
        self.liner2_1 = nn.Linear(256, 128)
        self.liner3_1 = nn.Linear(128, 1)

        self.liner1_2 = nn.Linear(16 * args.edge, 256)
        self.liner2_2 = nn.Linear(256, 128)
        self.liner3_2 = nn.Linear(128, 1)

        self.liner1_3 = nn.Linear(16 * args.edge, 256)
        self.liner2_3 = nn.Linear(256, 128)
        self.liner3_3 = nn.Linear(128, 1)

        self.liner1_4 = nn.Linear(16 * args.edge, 256)
        self.liner2_4 = nn.Linear(256, 128)
        self.liner3_4 = nn.Linear(128, 1)

        self.liner1_5 = nn.Linear(16 * args.edge, 256)
        self.liner2_5 = nn.Linear(256, 128)
        self.liner3_5 = nn.Linear(128, 1)

        self.liner1_6 = nn.Linear(16 * args.edge, 256)
        self.liner2_6 = nn.Linear(256, 128)
        self.liner3_6 = nn.Linear(128, 1)

        self.liner4_1 = nn.Linear(16 * args.edge, 256)
        self.liner5_1 = nn.Linear(256, 128)
        self.liner6_1 = nn.Linear(128, 1)

        self.liner4_2 = nn.Linear(16 * args.edge, 256)
        self.liner5_2 = nn.Linear(256, 128)
        self.liner6_2 = nn.Linear(128, 1)

        self.liner4_3 = nn.Linear(16 * args.edge, 256)
        self.liner5_3 = nn.Linear(256, 128)
        self.liner6_3 = nn.Linear(128, 1)

        self.liner4_4 = nn.Linear(16 * args.edge, 256)
        self.liner5_4 = nn.Linear(256, 128)
        self.liner6_4 = nn.Linear(128, 1)

        self.liner4_5 = nn.Linear(16 * args.edge, 256)
        self.liner5_5 = nn.Linear(256, 128)
        self.liner6_5 = nn.Linear(128, 1)

        self.liner4_6 = nn.Linear(16 * args.edge, 256)
        self.liner5_6 = nn.Linear(256, 128)
        self.liner6_6 = nn.Linear(128, 1)

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = self.graphu(x, edge_index)
        x = x.flatten()

        x11 = self.liner3_1(torch.sigmoid(self.liner2_1(torch.sigmoid(self.liner1_1(x)))))
        x21 = self.liner3_2(torch.sigmoid(self.liner2_2(torch.sigmoid(self.liner1_2(x)))))
        x31 = self.liner3_3(torch.sigmoid(self.liner2_3(torch.sigmoid(self.liner1_3(x)))))
        x41 = self.liner3_4(torch.sigmoid(self.liner2_4(torch.sigmoid(self.liner1_4(x)))))
        x51 = self.liner3_5(torch.sigmoid(self.liner2_5(torch.sigmoid(self.liner1_5(x)))))
        x61 = self.liner3_6(torch.sigmoid(self.liner2_6(torch.sigmoid(self.liner1_6(x)))))

        x12 = self.liner6_1(torch.sigmoid(self.liner5_1(torch.sigmoid(self.liner4_1(x)))))
        x22 = self.liner6_2(torch.sigmoid(self.liner5_2(torch.sigmoid(self.liner4_2(x)))))
        x32 = self.liner6_3(torch.sigmoid(self.liner5_3(torch.sigmoid(self.liner4_3(x)))))
        x42 = self.liner6_4(torch.sigmoid(self.liner5_4(torch.sigmoid(self.liner4_4(x)))))
        x52 = self.liner6_5(torch.sigmoid(self.liner5_5(torch.sigmoid(self.liner4_5(x)))))
        x62 = self.liner6_6(torch.sigmoid(self.liner5_6(torch.sigmoid(self.liner4_6(x)))))

        out = torch.cat([x11, x21, x31, x41, x51, x61,
                         x12, x22, x32, x42, x52, x62], dim=0).view(12, -1)

        return out

def data_iter(features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)                # 样本的读取顺序是随机的
    for i in range(0, num_examples):
        yield features[i], labels[i]

def logcosh(true, pred):        #计算真实标签 true 和预测标签 pred 之间的 log-cosh 损失
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)

log_var_a = torch.zeros((1,)).cuda()
log_var_a.requires_grad = True
log_var_b = torch.zeros((1,)).cuda()
log_var_b.requires_grad = True
log_var_c = torch.zeros((1,)).cuda()
log_var_c.requires_grad = True
log_var_d = torch.zeros((1,)).cuda()
log_var_d.requires_grad = True
log_var_e = torch.zeros((1,)).cuda()
log_var_e.requires_grad = True
log_var_f = torch.zeros((1,)).cuda()
log_var_f.requires_grad = True

log_var_a2 = torch.zeros((1,)).cuda()
log_var_a2.requires_grad = True
log_var_b2 = torch.zeros((1,)).cuda()
log_var_b2.requires_grad = True
log_var_c2 = torch.zeros((1,)).cuda()
log_var_c2.requires_grad = True
log_var_d2 = torch.zeros((1,)).cuda()
log_var_d2.requires_grad = True
log_var_e2 = torch.zeros((1,)).cuda()
log_var_e2.requires_grad = True
log_var_f2 = torch.zeros((1,)).cuda()
log_var_f2.requires_grad = True

# model = MTUNet(nfeat=7, nhid=256, nclass=64, dep=1, dropout=0.5, pool_ratios=0.5, sum_res=True).to(device)
model = MTUNet(nfeat=7, nhid=128, nclass=16, dep=2, dropout=0.5, pool_ratios=0.5, sum_res=True).to(device)

# loss_func = nn.MSELoss()
# optimizer = torch.optim.SGD([
#                 {'params': model.parameters()},
#                 {'params': awl.parameters()},
#                 {'params': awl2.parameters()}
#             ], lr=1e-3)

# get all parameters (model parameters + task dependent log variances)
params = ([p for p in model.parameters()] + [log_var_a] + [log_var_a2]
                                          + [log_var_b] + [log_var_b2]
                                          + [log_var_c] + [log_var_c2]
                                          + [log_var_d] + [log_var_d2]
                                          + [log_var_e] + [log_var_e2]
                                          + [log_var_f] + [log_var_f2])

# loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(params, lr=args.LR)

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 90], gamma=0.1)
train_loader = data_iter(idx_train, labels_train)
test_loader = data_iter(idx_test, labels_test)

def train(args, model, dataloader_train, dataloader_val):
    print("Train...")

    for epoch in range(args.EPOCH):
        model.train()
        train_loader = data_iter(idx_train, labels_train)

        for step, (x, y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            out = model(x, edge_index)

            pred1 = torch.cat((out[0], out[6]), 0)
            pred2 = torch.cat((out[1], out[7]), 0)
            pred3 = torch.cat((out[2], out[8]), 0)
            pred4 = torch.cat((out[3], out[9]), 0)
            pred5 = torch.cat((out[4], out[10]), 0)
            pred6 = torch.cat((out[5], out[11]), 0)

            lab1 = torch.cat((y[0], y[6]), 0)
            lab2 = torch.cat((y[1], y[7]), 0)
            lab3 = torch.cat((y[2], y[8]), 0)
            lab4 = torch.cat((y[3], y[9]), 0)
            lab5 = torch.cat((y[4], y[10]), 0)
            lab6 = torch.cat((y[5], y[11]), 0)

            # 计算损失值
            loss1 = criterion(pred1, lab1, [log_var_a, log_var_a2])
            loss2 = criterion(pred2, lab2, [log_var_b, log_var_b2])
            loss3 = criterion(pred3, lab3, [log_var_c, log_var_c2])
            loss4 = criterion(pred4, lab4, [log_var_d, log_var_d2])
            loss5 = criterion(pred5, lab5, [log_var_e, log_var_e2])
            loss6 = criterion(pred6, lab6, [log_var_f, log_var_f2])



            loss_all = loss1+ loss2+ loss3+ loss4+ loss5+ loss6
            # loss_all = awl7(loss_l1, loss_l2, loss_l3, loss_l4, loss_l5, loss_l6)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # lr_scheduler.step()

        # if epoch % args.validation_step == 0:
        val(model, test_loader, epoch)
#测试
def val(model, dataloader_val, epoch):
    print('Val...')
    start = time.time()
    with torch.no_grad():
        model.eval()

        MAEg_all = []
        MSEg_all = []
        Rg_all = []

        MAEb_all = []
        MSEb_all = []
        Rb_all = []


        test_loader = data_iter(idx_test, labels_test)

        for step, (x, y) in enumerate(test_loader):
            x,y = x.cuda(), y.cuda()
            predict = model(x, edge_index)      #model = MTUNet(nfeat=7, nhid=128, nclass=16, dep=2, dropout=0.5, pool_ratios=0.5, sum_res=True).to(device)
            label = y.squeeze()
            predict = predict.squeeze()

            labelg = label[0:6]
            labelb = label[6:]
            predictg = predict[0:6]
            predictb = predict[6:]


            maeg = MAE(predictg, labelg)
            mseg = MSE(predictg, labelg)
            rg = R(predictg, labelg)

            MAEg_all.append(maeg)
            MSEg_all.append(mseg)
            Rg_all.append(rg)

            maeb = MAE(predictb, labelb)
            mseb = MSE(predictb, labelb)
            rb = R(predictb, labelb)

            MAEb_all.append(maeb)
            MSEb_all.append(mseb)
            Rb_all.append(rb)

        maeg = np.mean(MAEg_all)
        mseg = np.mean(MSEg_all)
        rmseg = mseg ** (0.5)
        rg = np.mean(Rg_all)

        maeb = np.mean(MAEb_all)
        mseb = np.mean(MSEb_all)
        rmseb = mseb ** (0.5)
        rb = np.mean(Rb_all)

        str_ = ("%15.5g;" * 9) % (epoch + 1, maeg, mseg, rmseg, rg, maeb, mseb, rmseb, rb)
        with open('result_Uautoloss_K4_50_DD_DE_relu_SGD_Smooth.csv', 'a') as f:
            f.write(str_ + '\n')

        print('EPOCH:         {:}'.format(epoch+1))
        print('MAEg:          {:}'.format(maeg))
        print('MSEg:          {:}'.format(mseg))
        print('RMSEg:         {:}'.format(rmseg))
        print('Rg:            {:}'.format(rg))

        print('MAEb:          {:}'.format(maeb))
        print('MSEb:          {:}'.format(mseb))
        print('RMSEb:         {:}'.format(rmseb))
        print('Rb:            {:}'.format(rb))

        print('Time:         {:}s'.format(time.time() - start))
        print('*'*100)

        return None

train(args, model, train_loader, test_loader)
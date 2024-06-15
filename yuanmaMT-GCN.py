from __future__ import division
from __future__ import print_function
from torch_geometric.nn import HGTConv
import math
# from layer import TransformerModel
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
from torch_geometric.nn import JumpingKnowledge, SAGEConv, global_mean_pool
import torch
import torch.optim as optim
import torch.utils.data as Data
import random
import sys
# sys.path.append('F:\\Pycharm\\Projects\\SuChou')

# import pycocotools
import matplotlib.pyplot as plt
import pylab as pl
import time

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()#设置默认参数
parser.add_argument('--no-cuda',         action='store_true',  default=False,   help='Disables CUDA training.')
parser.add_argument('--fastmode',        action='store_true',  default=False,   help='Validate during training pass.')
parser.add_argument('--EPOCH',           type=int,             default=100,     help='Number of epochs to train.')
parser.add_argument('--LR',              type=float,           default=0.0002,  help='Initial learning rate.')
parser.add_argument('--weight_decay',    type=float,           default=5e-4,    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--validation_step', type=int,             default=1,       help='check model for every X time')
parser.add_argument('--hidden',          type=int,             default=256,     help='Number of hidden units.')
parser.add_argument('--dropout',         type=float,           default=0.5,     help='Dropout rate (1 - keep probability).')
parser.add_argument('--edge',            type=float,           default=6,       help='numbers of sides.')
parser.add_argument('--mu',              type=float,           default=0,       help='Gauss noise mean.')#高斯噪声的均值。默认是 0。
parser.add_argument('--sigma',           type=float,           default=5,       help='Gauss noise std.')#高斯噪声的标准差。默认是 5。
args = parser.parse_args()#获取所有的命令行参数，并将它们保存在args对象中
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#data preprocess
df0 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\00春古.csv')
df1 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\01春越.csv')
df2 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\02古江.csv')
df3 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\03古舜.csv')
df4 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\04萧越.csv')
df5 = pd.read_csv('C:\\Users\\ASUS\\Desktop\\图神经代码及论文\\05萧古.csv')
#使用 pandas 的 read_csv 函数从指定路径读取了六个 CSV 文件，并将它们分别赋值给 df0 到 df5

df01 = df0.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))#应用lambda函数进行标准化处理。这个lambda函数将每一列的值减去该列的均值，然后除以该列的标准差。
df11 = df1.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df21 = df2.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df31 = df3.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df41 = df4.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
df51 = df5.iloc[:,1:8].apply(lambda x:((x-x.mean())/x.std()))
# 这个处理的结果被分别保存在 df01 到 df51 中，所以这些 DataFrame 现在包含了原始数据的标准化版本，这样在进行机器学习训练时，不会有某个特征由于其数值范围大而主导整个模型。

# DataFrame 是一个来自 pandas 库的数据结构。它是一种二维表格型数据结构，其中数据以行和列的形式进行组织。
# DataFrame 是一个非常灵活的数据结构，可以用来处理不同类型的数据，如数值、字符串、时间序列等。
# DataFrame 提供了许多方便的数据操作和分析功能，如数据清洗、排序、分组、聚合、统计分析等，可以大大简化数据处理和分析的任务。

df0 = pd.concat([df01,df0.iloc[:,8:10]],axis=1)
df1 = pd.concat([df11,df1.iloc[:,8:10]],axis=1)
df2 = pd.concat([df21,df2.iloc[:,8:10]],axis=1)
df3 = pd.concat([df31,df3.iloc[:,8:10]],axis=1)
df4 = pd.concat([df41,df4.iloc[:,8:10]],axis=1)
df5 = pd.concat([df51,df5.iloc[:,8:10]],axis=1)
#这段代码的目的是将之前标准化处理后的数据（df01, df11, df21, df31, df41, df51）与原始数据中的第9列和第10列（未经标准化处理）合并在一起。
# pd.concat（）函数用于沿指定轴（axis）连接两个或多个 DataFrame。
# print(df0.shape)  (8640,9)
# print(df0.shape[0])

df0 = np.array(df0)#将之前的 DataFrame 数据df0, df1, df2, df3, df4, df5转换为 NumPy 数组（使用np.array（）函数）
df1 = np.array(df1)
df2 = np.array(df2)
df3 = np.array(df3)
df4 = np.array(df4)
df5 = np.array(df5)
#定义储存空间，是一个三维列表，用于存储特征数据。
features = [[[[]for p in range(7)]for i in range(6)]for j in range(8640)]
# 数组或列表的索引通常从最外层开始，第一维度是最外层：8640个样本个数（长），第二维度是下一层：6个数据集（宽），第三维度是下一层：7个特征（高、深度）。
# 第一维度、第二维度、第三维度可以类比为立方体的三个维度：长、宽、高。或者说行、列、深。
# 不过在实际的数据处理和编程中，这些 "维度" 更多的是逻辑上的组织方式，而不是物理空间的长、宽、高（深度）。

labels1 = [[[[]for p in range(1)]for i in range(6)]for j in range(8640)]
labels2 = [[[[]for p in range(1)]for i in range(6)]for j in range(8640)]
#labels1 和 labels2: 这两个是三维列表，用于存储两种不同的标签数据。都有 8640 个样本，6 个数据集， 1 个标签。
# 标签是我们想要预测的目标。在监督学习中，标签是已知的输出。

for row in range(0,df0.shape[0]):#df0.shape[0]获取的是df0数组的行数（也就是样本数量）。
    #这个for循环，range(0, df0.shape[0]) 表示的是遍历从第一行到最后一行的所有行。
    for j in range(0,7):#有三个中括号的原因：该数据为三维数据，（8640，6，7），中间的数据是数据集的个数，所以从0到5。
        features[row][0][j] = df0[row][:][j]#df0[row][:][j] 这部分获取的是第 row 行，第 j 列的数据。
        #其中row是在外层循环中定义的，它遍历所有的行。这里的row会从0遍历到df0.shape[0] - 1，也就是df0数组的行数减一，这样就遍历了所有的样本
        features[row][1][j] = df1[row][:][j]
        features[row][2][j] = df2[row][:][j]
        features[row][3][j] = df3[row][:][j]
        features[row][4][j] = df4[row][:][j]
        features[row][5][j] = df5[row][:][j]
#这个循环将每个 DataFrame 的前 7 列（特征）填充到 features 结构中。这里有两个嵌套的 for 循环。外部循环遍历所有样本，内部循环遍历每个样本的 7 个特征。
for row in range(0,df0.shape[0]):
    for j in range(1):
        labels1[row][0][j] = df0[row][:][j+7]
        labels1[row][1][j] = df1[row][:][j+7]
        labels1[row][2][j] = df2[row][:][j+7]
        labels1[row][3][j] = df3[row][:][j+7]
        labels1[row][4][j] = df4[row][:][j+7]
        labels1[row][5][j] = df5[row][:][j+7]
#第二个 for 循环（第二部分）：与第一个 for 循环类似，但这次是将每个 DataFrame 的第 8 列（索引为 j+7）填充到 labels1 结构中。
#这里，外部循环遍历所有样本，内部循环遍历每个样本的一个标签。
for row in range(0,df0.shape[0]):
    for j in range(1):
        labels2[row][0][j] = df0[row][:][j+8]
        labels2[row][1][j] = df1[row][:][j+8]
        labels2[row][2][j] = df2[row][:][j+8]
        labels2[row][3][j] = df3[row][:][j+8]
        labels2[row][4][j] = df4[row][:][j+8]
        labels2[row][5][j] = df5[row][:][j+8]
#第三个 for 循环（第三部分）：与第二个 for 循环类似，但这次是将每个 DataFrame 的第 9 列（索引为 j+8）填充到 labels2 结构中。
#这里，外部循环遍历所有样本，内部循环遍历每个样本的一个标签。
features = np.array(features,dtype=np.float32)#将嵌套列表features转换为Numpy数组，同时指定了数据类型为np.float32。
features = np.nan_to_num(features)#np.nan_to_num函数替换features中的NaN（缺失）值，将它们替换为0（处理缺失值）。这可以确保在后续计算中不会出现NaN值引起的问题。
labels1 = np.array(labels1,dtype=np.float32)
labels2 = np.array(labels2,dtype=np.float32)
#将labels1、labels2 转换为 NumPy 数组，并将它们的数据类型设置为 np.float32。

# print(features.shape)
# print(labels1.shape)
# print(labels2.shape)

#加入高斯噪声：这种噪声可以看作是一种数据增强技术。通过向输入数据中引入一定程度的随机性，模型被迫学习更为鲁棒的特征，从而提高泛化能力/防止过拟合。
def gen_gaussian_noise(signal, SNR): #函数的作用是根据给定的信号和信噪比（SNR）生成高斯噪声
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比：信号的功率与背景噪声的功率之比。SNR越高，信号质量越好。
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape) #*signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1/signal.shape[0]) * np.sum(np.power(signal,2))
    noise_variance = signal_power/np.power(10, (SNR/10))
    #noise_variance计算了噪声的方差（即噪声功率），它是通过将信号的功率除以一个由信噪比确定的值得到的。如果信噪比越高，计算出的噪声方差就会越小，生成的噪声就会越弱。
    noise = (np.sqrt(noise_variance) / np.std(noise))*noise
    return noise
#丢失节点
# Drop Edge
# N = 1 drop edge
for i in range(8640):
    for j in range(6):
        ssp = random.random()#对于features中的个样本的每个数据集（总共 6 个）进行操作，生成一个介于 0 和 1 之间的随机数 ssp
        ssr = float(1/6)
        if ssp < ssr:
            features[i][j] = 0#如果这个随机数小于 1/6，那么就将对应的 features 元素设置为零
#这段代码的主要目标是以一定的概率随机地将 features 中的某些元素设置为零。这个过程通常被称为"dropout"，它是一种在深度学习中常用的正则化技术，用于防止模型过拟合。
#丢失数据，即节点特征
# Drop Data
# N = 0.01 drop data
for i in range(8640):
    for j in range(6):
        for p in range(7):
            sd = random.random()#random.random()是random模块中的一个函数，它返回一个随机生成的浮点数，范围在[0,1)
            sf = 0.01
            if sd < sf:
                features[i][j][p] = 0
#这里的代码遍历了features的每一个元素（到p。针对第三维度的特征部分）。
# 对于每一个元素，它生成一个[0,1)之间的随机数sd，如果sd小于0.01（sf），那么对应的特征就会被置为0，即删除该特征。

features = torch.FloatTensor(features)
labels1 = torch.FloatTensor(labels1)
labels2 = torch.FloatTensor(labels2)
#将NumPy数组转换为PyTorch的张量数据类型，以便在PyTorch框架中使用。
#torch.FloatTensor()是用于将数据转换为浮点类型张量的函数。features,labels1,labels2都被转换为了FloatTensor类型，以便之后用于训练模型。

#划分数据集为训练集、测试集1、测试集2
idx_train = features[0:6912, :]
labels1_train = labels1[0:6912,:]
#idx_train和labels1_train：从features和labels1的开始到6912（不包括6912）的数据，用作训练集。
idx_test2 = features[6912:7776,:]
labels1_test2 = labels1[6912:7776,:]
#idx_test2和labels1_test2：从features和labels1的6912到7776（不包括7776）的数据，用作第一组测试集。
idx_test = features[7776:8640,:]
labels1_test = labels1[7776:8640,:]
#idx_test和labels1_test：从features和labels1的7776到8640（不包括8640）的数据，用作第二组测试集。

#这个函数定义的是平均绝对误差(Mean Absolute Error,MAE)的计算方法
def MAE(SR, GT):
    SR = SR.flatten()#将预测值SR展平为一维张量。目的是为了保证预测值和实际值有相同的形状，以便于后续计算他们之间的差值。
    GT = GT.flatten()#将实际值GT展平为一维张量。
    corr = torch.sum(abs(SR - GT))#计算预测值和实际值之间的绝对偏差，并求和。
    tensor_size = SR.size(0)#计算预测值的元素个数。
    mae = float(corr) / float(tensor_size)#计算平均绝对误差（绝对偏差的总和除以元素的个数）。
    return mae

#这个函数是用来计算平均平方误差(Mean Squared Error,MSE)的计算方法
def MSE(SR, GT):
    SR = SR.flatten()#同上
    GT = GT.flatten()#同上
    corr = torch.sum((SR - GT)**2)#这一行是计算预测值和实际值之间的平方差并求和。与MAE计算区别：MAE计算的是绝对差，而MSE计算的是平方差。
    tensor_size = SR.size(0)#同上
    mse = float(corr) / float(tensor_size)#计算平均平方误差（将平方差的总和除以元素的个数）。
    return mse

#计算决定系数R^2，它衡量的是模型预测的方差占总方差的比例
def R(SR, GT):
    cov = torch.cov(torch.stack((SR, GT)))[0, 1]
    var_1 = torch.var(SR)
    var_2 = torch.var(GT)
    std_1 = torch.sqrt(var_1)
    std_2 = torch.sqrt(var_2)
    r = float(cov) / (float(std_1) * float(std_2))
    #corr = MSE(SR, GT)#调用前面定义的MSE函数，计算预测值SR和实际值GT之间的平均平方误差。
    #varr = torch.var(GT)#计算实际值GT的方差
    #r = 1 - float(corr)/float(varr)#计算决定系数 R^2（用1减去预测的方差占总方差的比例）。
    return r

def normalize(mx):
    #定义了一个normalize函数，对输入矩阵mx进行归一化操作，并对邻接矩阵adj进行归一化处理。该段作用在241行体现，D^-1A
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))#计算输入矩阵mx每一行的和并储存在rowsum中
    r_inv = np.power(rowsum, -1).flatten()#计算rowsum的倒数并储存在r_inv中
    r_inv[np.isinf(r_inv)] = 0. #将r_inv中无穷大的数用0替代
    r_mat_inv = sp.diags(r_inv)#将r_inv转化为对角矩阵r_mat_inv，且对角元素即为r_inv的值
    mx = r_mat_inv.dot(mx)#对角阵与mx相乘，得到归一化后的矩阵。
    return mx
# mx 在这个函数中代表的是任何需要进行行归一化的矩阵（也就是说 mx 可以是特征矩阵 X，也可以是邻接矩阵 A 或者其它类型的矩阵）。
# 这里操作可以对标D ̃ ^-1/2 A ̃ D ̃ ^-1/2这个经典的归一化公式（但并不完全一样，normalize只是简单地进行了行归一化，即让每一行的和为1，这是一种更简单的归一化方法。）

adj = np.array([[0., 1., 1., 1., 1., 1.], #定义了一个6x6的邻接矩阵adj（常说的A），代表一个六节点的完全图，即任意两个顶点之间都有边。
                [1., 0., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1., 1.],
                [1., 1., 1., 0., 1., 1.],
                [1., 1., 1., 1., 0., 1.],
                [1., 1., 1., 1., 1., 0.]])
adj = adj + np.multiply(adj.T,(adj.T > adj)) - np.multiply(adj,(adj.T > adj))#将adj的转置与adj相加，确保 adj 是一个对称矩阵。
# adj = normalize(adj+np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float)) #进行邻接矩阵的正则化操作

adj = normalize(adj)# 使用normalize函数对邻接矩阵adj进行归一化处理
adj = torch.FloatTensor(adj)# 将 numpy 数组 adj 转化为 PyTorch 的张量（Tensor）形式，因为 PyTorch 的运算需要使用 Tensor 进行
adj = adj.cuda()# 将数据移动到 GPU 上


class GraphConvolution(Module):#定义GraphConvolution（一个GCN卷积层）这个类
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907（这个网址就是github上GCN卷积层的论文描述）
    """
    def __init__(self, in_features, out_features, bias=True):#定义GraphConvolution类中的__init__方法，括号内的in_features, out_features, bias=True是属性
        # in_features 和 out_features 分别代表输入特征和输出特征的数量。在构建一个更复杂的模型时（如这个 GCN 模型），通常会将这些参数重命名，以适应更具体的模型结构和任务需求。
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features#这个类的构造函数 __init__ 接收输入特征的数量(in_features)和输出特征的数量(out_features)作为参数
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#创建一个权重矩阵(self.weight)，这个矩阵的大小是 in_features x out_features
        # 比如说 in_features 是5，out_features 是3，那么 self.weight 就是一个5x3的二维张量（或者说是一个5x3的矩阵）
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#如果 bias 参数为 True，那么还会创建一个数值大小为 out_features 的偏置向量(self.bias)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()#调用self.reset_parameters()来初始化权重和偏置参数。

    def reset_parameters(self):# 定义GraphConvolution类中的 reset_parameters 方法：用于初始化权重和偏置参数
        #初始化权重原因：self.weight（权重矩阵）中的每个元素都是一个浮点数，初始时被随机初始化（就是在这里），然后在训练过程中被优化
        stdv = 1. / math.sqrt(self.weight.size(1))
        #计算了一个名为 stdv 的标准差。这个值是根据权重矩阵的列数（self.weight.size(1)）计算的。它是权重矩阵列数的倒数平方根的倒数。
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        #使用均匀分布 uniform_() 进行初始化，范围是从 -stdv 到 stdv，初始化了权重矩阵（self.weight.data）和偏置向量（self.bias.data）
        #这样的初始化方法可以使得权重和偏置参数在初始阶段具有相对较小的值，有助于网络的训练和收敛。

    def forward(self, input, adj):# 定义GraphConvolution类中的 forward 方法：定义了图卷积的前向传播过程（图卷积操作的核心部分）：通过输入特征input、权重矩阵以及邻接矩阵adj来计算输出特征output
        support = torch.mm(input, self.weight)#先通过矩阵乘法 torch.mm(input, self.weight) 计算输入特征矩阵 input 和权重矩阵 self.weight 的乘积，得到 support
        output = torch.mm(adj, support)#再通过矩阵乘法 torch.mm(adj, support) ，计算 support 与邻接矩阵 adj 相乘，得到输出特征 output
        if self.bias is not None:#如果存在偏置参数 self.bias=True ，那么将偏置向量加到输出特征矩阵 output 上，并返回这个值
            return output + self.bias
        else:
            return output

    def __repr__(self):# 定义GraphConvolution类中的 __repr__ 方法：用于提供类的字符串表示（当要打印一个对象时，Python会调用这个方法来获取要打印的字符串）。
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    # 这里的 __repr__ 方法会返回一个形如 "GraphConvolution (in_features -> out_features)" 的字符串。
    # 比如in_features是10，out_features是20，那么可以得到结果GraphConvolution (10 -> 20)。
    # 也就是说这个__repr__方法其实是帮助阅读代码者理解的，对模型本身没有提供什么功能。__repr__方法主要是用于开发和调试

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):#这个__init__方法定义了模型的主要组成部分
        super(GCN, self).__init__()#首先利用之前定义的GraphConvolution类创建了两个图卷积层（self.gcn1和self.gcn2）
        # GCN模块的体现：
        self.gcn1 = GraphConvolution(nfeat,nhid)#第一个图卷积层接收输入特征（数量由nfeat参数给出），并输出到隐藏层（隐藏层的节点数由nhid参数给出）
        self.gcn2 = GraphConvolution(nhid,nclass)#第二个图卷积层接收这些隐藏层的输出，并产生最终的输出特征（数量由nclass参数给出）
        # nfeat 相当于 gcn1 的 in_features，nhid 相当于 gcn1 的 out_features 和 gcn2 的 in_features，nclass 相当于 gcn2 的 out_features.
        # 这对于 in_features，out_features 是重命名了。这种参数命名的方式主要是为了在模型的更高层级上提供有关模型结构和功能的信息。比如 nfeat 意味着 "number of features"，nhid 意味着 "number of hidden units"，nclass 意味着 "number of classes"
        # 当我们在定义模型的结构时（在 __init__ 函数中），我们会创建 GraphConvolution 层的实例，并用 nfeat 等参数来指定输入特征和输出特征的数量。
        # 这样的话，当这个 GraphConvolution 层被调用（在 forward 函数中）时，它就会知道期望的输入特征矩阵的大小，以及期望的输出特征矩阵的大小。

        # Multi-Task FCN模块的体现：
        self.liner1_1 = nn.Linear(64 * args.edge, 256)#第一层线性变换，将GCN的输出（假设为64*args.edge维）转换为256维的特征表示
        self.liner1_2 = nn.Linear(256,128)#第二层线性变换，将256维的特征表示转换为128维。
        self.liner1_3 = nn.Linear(128,1)#第三层线性变换，将128维的特征表示转换为1维，这1维就是特定任务的预测输出。

        self.liner2_1 = nn.Linear(64 * args.edge, 256)
        self.liner2_2 = nn.Linear(256, 128)
        self.liner2_3 = nn.Linear(128, 1)

        self.liner3_1 = nn.Linear(64 * args.edge, 256)
        self.liner3_2 = nn.Linear(256, 128)
        self.liner3_3 = nn.Linear(128, 1)

        self.liner4_1 = nn.Linear(64 * args.edge, 256)
        self.liner4_2 = nn.Linear(256, 128)
        self.liner4_3 = nn.Linear(128, 1)

        self.liner5_1 = nn.Linear(64 * args.edge, 256)
        self.liner5_2 = nn.Linear(256, 128)
        self.liner5_3 = nn.Linear(128, 1)

        self.liner6_1 = nn.Linear(64 * args.edge, 256)
        self.liner6_2 = nn.Linear(256, 128)
        self.liner6_3 = nn.Linear(128, 1)
        # 模型还定义了六组全连接层，每组包含三个线性层。这些线性层用于进一步处理通过图卷积层得到的特征。
        # 每组 linerX_1、linerX_2 和 linerX_3 定义了一个全连接网络（FCN）模块，其中X表示任务的编号，这里定义了六个不同的任务（六个模块），这些网络模块在整个模型中担当多任务处理的角色。
        # 每个任务（每个模块）的全连接网络又由三个线性层组成，这三个线性层负责将GCN的输出转换为特定任务的预测输出。

        self.dropout = dropout# 定义了一个dropout参数，它在前向传播过程中被用于正则化

    def forward(self,x):# 定义了模型的前向传播过程
        x = torch.relu(self.gcn1(x, adj))# 将输入特征x和邻接矩阵adj传递给第一个图卷积层，并通过ReLU激活函数处理输出为x
        x = self.gcn2(x,adj)# 将结果x和邻接矩阵传递给第二个图卷积层，输出为x
        x = x.flatten()# 将结果x展平（flatten）成一维的向量。这是因为全连接层（Linear）期望的输入是一维的向量。
        # 这部分的代码实际上在调用在__init__方法中定义的图卷积层（293 294行）
        # self.gcn1(x, adj)和self.gcn2(x, adj)是通过调用GraphConvolution类的forward方法，以实现图卷积的操作。这个操作实质上是将邻接矩阵和输入特征相乘，以在节点间传递信息

        x11 = self.liner1_3(torch.sigmoid(self.liner1_2(torch.relu(self.liner1_1(x)))))
        #比如说，针对第一个任务：x传入liner1_1，经过relu，再传入liner1_2，再经过sigmoid，最后传入liner1_3。走完第一组全连接模块（对应第一个任务），输出第一个任务的预测结果x11（是向量）。
        x21 = self.liner2_3(torch.sigmoid(self.liner2_2(torch.relu(self.liner2_1(x)))))
        x31 = self.liner3_3(torch.sigmoid(self.liner3_2(torch.relu(self.liner3_1(x)))))
        x41 = self.liner4_3(torch.sigmoid(self.liner4_2(torch.relu(self.liner4_1(x)))))
        x51 = self.liner5_3(torch.sigmoid(self.liner5_2(torch.relu(self.liner5_1(x)))))
        x61 = self.liner6_3(torch.sigmoid(self.liner6_2(torch.relu(self.liner6_1(x)))))
        # 对于每一个任务（总共有六个），将这个一维向量通过一系列的全连接层（Linear）和激活函数（ReLU、Sigmoid）进行处理。这个过程生成了六个输出向量，每个向量对应一个任务的预测结果。
        # 这里的x被用六遍，每一次被用于生成一个不同任务的预测结果。
        # 每个liner系列的输出形状应该是(batch_size, 1)，其中batch_size是输入到模型的数据批次的大小。
        out = torch.cat([x11,x21,x31,x41,x51,x61],dim=0).view(6,-1)# 将这些输出向量在第0维度上拼接为一个形状为[6,batch_size]的张量。这个张量的每一行都包含了一个任务的预测结果。
        # torch.cat 是一个将多个张量按指定的维度拼接起来的函数，view 是一个改变张量形状的函数，-1 表示该维度的大小由其他维度的大小决定（也就是batch_size）。
        return out
        # 返回这个每一行都包含了一个任务的预测结果的张量out。

def data_iter(features,labels):# 定义了一个数据迭代器，用于在训练和测试过程中提供数据
    num_examples = len(features)
    indices = list(range(num_examples))# 生成一个包含所有样本索引的列表
    random.shuffle(indices)# 对这个列表进行随机打乱
    for i in range(0,num_examples):
        yield features[i],labels[i]
    # 在每次迭代时，返回一个包含特征和标签的样本

# 这部分代码用于创建训练和测试的数据加载器、定义模型、优化器和损失函数。
train_loader = data_iter(idx_train,labels1_train)# idx_train是训练集的特征矩阵，labels1_train是对应的训练集标签。这些数据将被传递给data_iter函数，在每次迭代时生成一个训练样本。
test_loader = data_iter(idx_test,labels1_test)# idx_test是测试集的特征矩阵，labels1_test是对应的测试集标签。它们也将被传递给data_iter函数，用于生成测试样本。
# 调用data_iter函数创建训练数据加载器train_loader和测试数据加载器test_loader。这些加载器会在每次迭代时返回一个样本。

model = GCN(nfeat=7,nhid=512,nclass=64,dropout=0.5).cuda()# 创建了GCN模型的一个实例，并将模型移动到GPU上
optimizer = torch.optim.Adam(model.parameters(),lr=args.LR)# 定义了一个优化器（这里使用的是Adam算法），并设置了学习率
loss_func = nn.MSELoss()# 定义了一个损失函数（这里使用的是均方误差损失）。

def train(args,model,optimizer,dataloader_train,dataloader_val):# 这行定义了一个名为train的函数，它接收5个参数
    # args：一个包含训练参数（例如，训练周期数、学习率等）的对象。
    # model：待训练的模型。
    # optimizer：用于更新模型参数的优化器。
    # dataloader_train：一个加载训练数据的迭代器。
    # dataloader_val：一个加载验证数据的迭代器。
    print('Train...')# 用于在训练过程开始时在控制台上打印一条信息，以提示当前正在进行训练阶段。这样可以提供训练过程的可视化和进度追踪
    for epoch in range(args.EPOCH):# 这行代码开始了一个循环，循环次数等于训练周期数。每个训练周期，模型都会看到所有的训练数据。
        model.train()# 将模型设置为训练模式
        train_loader = data_iter(idx_train,labels1_train)# 创建一个数据迭代（加载）器 train_loader，用于在训练过程中按批次提供训练数据（用的）。
        for step, (x,y) in enumerate(train_loader):# 这行代码开始了一个内部循环，用于遍历训练数据。通过enumerate(train_loader)遍历训练数据集，获取每个训练样本的特征 x 和标签 y。
            x, y = x.cuda(),y.cuda() # 将数据移动到GPU设备
            out = model(x)# 将特征 x 输入模型 model，得到输出 out
            loss_1 = 0# 这行代码初始化变量loss_1（初始值是0嘛），它用于累积6个任务的损失。

            for i in range(6): # 这行代码开始了另一个内部循环，用于计算6个任务的损失。
                loss = loss_func(out[i][0],y[i][0])# 这行代码计算第i个任务的损失值。
                loss_1 += loss# 这行代码将当前任务的损失值累加到loss_1。

            loss_1 /=6# 这行代码将loss_1除以任务数量（即6），得到平均损失值。
            optimizer.zero_grad()# 这行代码清零所有已经累积的梯度。在PyTorch中，梯度默认会累积，所以每次优化前需要清零。
            loss_1.backward()# 这行代码通过反向传播计算梯度
            optimizer.step()#  这行代码使用计算得到的梯度更新模型参数。

        val(model,test_loader,epoch)# 这行代码在每个训练周期结束后，在验证集上评估模型的性能。


def val(model,dataloader_val,epoch):# val函数用于在验证集上评估模型性能
    print('Val...')# 用于在评估过程开始时在控制台上打印一条信息，以提示当前正在进行评估模型（验证）阶段。
    start = time.time()# 记录了开始的时间
    with torch.no_grad():# 这个语句用于暂停所有的梯度计算。因为在验证阶段，我们不需要更新模型参数，所以也就不需要计算梯度。
        model.eval()# 将模型调到评估模式

        MAE_all = []
        MSE_all = []
        R_all = []
        test_loader2 = data_iter(idx_test2, labels1_test2)# 创建一个数据迭代（加载）器 train_loader2，用于在评估过程中按批次提供验证数据。
        for step, (x, y) in enumerate(test_loader2):# 用来遍历验证集的所有样本。
            x, y = x.cuda(), y.cuda()# 将数据移动到GPU
            predict = model(x)# 输入数据x传递给模型，获取模型的输出，即预测值
            label = y.squeeze()
            predict = predict.squeeze()
            print(predict)
            mae = MAE(predict, label)
            mse = MSE(predict, label)
            r = R(predict, label)# 这几行代码是计算模型预测结果predict与真实标签label之间的平均绝对误差(MAE)，均方误差(MSE)和R值
            MAE_all.append(mae)
            MSE_all.append(mse)
            R_all.append(r)# 这几行代码将当前样本的MAE，MSE，R值添加到相应的列表中，以便后面计算平均值。

        mae = np.mean(MAE_all)
        mse = np.mean(MSE_all)
        rmse = mse ** (0.5)
        r = np.mean(R_all)# 这几行代码计算了所有样本的平均MAE，MSE和R值。

        str_ = ("%15.5g;" * 5) % (epoch + 1, mae, mse, rmse, r)# 格式化字符串，包含了当前的训练周期和各项指标的值。
        with open('GCN_2_b_LabelNoNoise.csv', 'a') as f:
            f.write(str_ + '\n')#这两行代码将这些指标写入一个csv文件。

        print('EPOCH2:        {:}'.format(epoch + 1))
        print('MAE2:          {:}'.format(mae))
        print('MSE2:          {:}'.format(mse))
        print('RMSE2:         {:}'.format(rmse))
        print('R2:            {:}'.format(r))
        print('Time2:         {:}s'.format(time.time() - start))
        print('*' * 100)#最后的几行代码是打印相关的指标和验证阶段的时间，并打印一行星号作为分隔符

        return None
train(args, model, optimizer, train_loader, test_loader)
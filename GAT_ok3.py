import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import random
import argparse
import time
import scipy.sparse as sp
from torch.nn.modules.module import Module
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--EPOCH', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--LR', type=float, default=0.00002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--validation_step', type=int, default=1, help='check model for every X time')
parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--edge', type=float, default=6, help='numbers of sides.')
parser.add_argument('--mu', type=float, default=0, help='Gauss noise mean.')
parser.add_argument('--sigma', type=float, default=5, help='Gauss noise std.')
args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
df0 = np.array(df0)
df1 = np.array(df1)
df2 = np.array(df2)
df3 = np.array(df3)
df4 = np.array(df4)
df5 = np.array(df5)
features = [[[[]for p in range(7)]for i in range(6)]for j in range(8640)]
labels1 = [[[[]for p in range(1)]for i in range(6)]for j in range(8640)]
labels2 = [[[[]for p in range(1)]for i in range(6)]for j in range(8640)]
for row in range(0, df0.shape[0]):
    for j in range(0, 7):
        features[row][0][j] = df0[row][:][j]
        features[row][1][j] = df1[row][:][j]
        features[row][2][j] = df2[row][:][j]
        features[row][3][j] = df3[row][:][j]
        features[row][4][j] = df4[row][:][j]
        features[row][5][j] = df5[row][:][j]
for row in range(0, df0.shape[0]):
    for j in range(1):
        labels1[row][0][j] = df0[row][:][j+7]
        labels1[row][1][j] = df1[row][:][j+7]
        labels1[row][2][j] = df2[row][:][j+7]
        labels1[row][3][j] = df3[row][:][j+7]
        labels1[row][4][j] = df4[row][:][j+7]
        labels1[row][5][j] = df5[row][:][j+7]
for row in range(0, df0.shape[0]):
    for j in range(1):
        labels2[row][0][j] = df0[row][:][j+8]
        labels2[row][1][j] = df1[row][:][j+8]
        labels2[row][2][j] = df2[row][:][j+8]
        labels2[row][3][j] = df3[row][:][j+8]
        labels2[row][4][j] = df4[row][:][j+8]
        labels2[row][5][j] = df5[row][:][j+8]
features = np.array(features, dtype=np.float32)
features = np.nan_to_num(features)
labels1 = np.array(labels1, dtype=np.float32)
labels2 = np.array(labels2, dtype=np.float32)
features = torch.FloatTensor(features)
labels1 = torch.FloatTensor(labels1)
labels2 = torch.FloatTensor(labels2)
idx_train = features[0:6912, :]
labels1_train = labels1[0:6912, :]
idx_test2 = features[6912:7776, :]
labels1_test2 = labels1[6912:7776, :]
idx_test = features[7776:8640, :]
labels1_test = labels1[7776:8640, :]
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
    cov = torch.cov(torch.stack((SR, GT)))[0, 1]
    var_1 = torch.var(SR)
    var_2 = torch.var(GT)
    std_1 = torch.sqrt(var_1)
    std_2 = torch.sqrt(var_2)
    r = float(cov) / (float(std_1) * float(std_2))
    return r
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
adj = np.array([[0., 1., 1., 1., 1., 1.],
                [1., 0., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1., 1.],
                [1., 1., 1., 0., 1., 1.],
                [1., 1., 1., 1., 0., 1.],
                [1., 1., 1., 1., 1., 0.]])
adj = adj + np.multiply(adj.T, (adj.T > adj)) - np.multiply(adj, (adj.T > adj))
adj = normalize(adj)
adj = torch.FloatTensor(adj)
adj = adj.cuda()
# Define the GAT layer
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        self.a = Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        h = h.view(N, -1)
        a_input = torch.cat([h.repeat(N, 1).view(N * N, -1), h.repeat(1, N).view(-1, N * N).T], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)
        output = torch.relu(h_prime)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
# Define the GAT model
class GAT(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions1 = GraphAttentionLayer(nfeat, nhid1, dropout=dropout, alpha=alpha)
        self.attentions2 = GraphAttentionLayer(nhid1, nhid2, dropout=dropout, alpha=alpha)
        self.out_att = GraphAttentionLayer(nhid2, nclass, dropout=dropout, alpha=alpha)
        self.liner1_1 = nn.Linear(64 * args.edge, 256)
        self.liner1_2 = nn.Linear(256, 128)
        self.liner1_3 = nn.Linear(128, 1)
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
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.attentions1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.attentions2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.out_att(x, adj))
        x = x.flatten()
        x11 = self.liner1_3(torch.sigmoid(self.liner1_2(torch.relu(self.liner1_1(x)))))
        x21 = self.liner2_3(torch.sigmoid(self.liner2_2(torch.relu(self.liner2_1(x)))))
        x31 = self.liner3_3(torch.sigmoid(self.liner3_2(torch.relu(self.liner3_1(x)))))
        x41 = self.liner4_3(torch.sigmoid(self.liner4_2(torch.relu(self.liner4_1(x)))))
        x51 = self.liner5_3(torch.sigmoid(self.liner5_2(torch.relu(self.liner5_1(x)))))
        x61 = self.liner6_3(torch.sigmoid(self.liner6_2(torch.relu(self.liner6_1(x)))))
        out = torch.cat([x11, x21, x31, x41, x51, x61], dim=0).view(6, -1)
        return out
def data_iter(features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples):
        yield features[i], labels[i]
train_loader = data_iter(idx_train, labels1_train)
test_loader = data_iter(idx_test, labels1_test)
model = GAT(nfeat=7,nhid1=256, nhid2=512, nclass=64, dropout=0.5, alpha=0.2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
loss_func = nn.MSELoss()
def train(args, model, optimizer):
    print('Train...')
    for epoch in range(args.EPOCH):
        model.train()
        train_loader = data_iter(idx_train, labels1_train)
        for step, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss_1 = 0
            for i in range(6):
                loss = loss_func(out[i][0], y[i][0])
                loss_1 += loss
            loss_1 /= 6
            optimizer.zero_grad()
            loss_1.backward()
            optimizer.step()
        if (epoch + 1) % args.validation_step == 0:
            validate(model, epoch)
def validate(model, epoch):
    print('Val...')
    start = time.time()
    with torch.no_grad():
        model.eval()
        MAE_all = []
        MSE_all = []
        R_all = []
        test_loader2 = data_iter(idx_test2, labels1_test2)
        for step, (x, y) in enumerate(test_loader2):
            x, y = x.cuda(), y.cuda()
            predict = model(x)
            label = y.squeeze()
            predict = predict.squeeze()
            mae = MAE(predict, label)
            mse = MSE(predict, label)
            r = R(predict, label)
            MAE_all.append(mae)
            MSE_all.append(mse)
            R_all.append(r)
        mae = np.mean(MAE_all)
        mse = np.mean(MSE_all)
        rmse = mse ** (0.5)
        r = np.mean(R_all)
        str_ = ("%15.5g;" * 6) % (epoch + 1, mae, mse, rmse, r, time.time() - start)
        with open('GAT_label1_no.csv', 'a') as f:
            f.write(str_ + '\n')
        print('EPOCH2:        {:}'.format(epoch + 1))
        print('MAE2:          {:}'.format(mae))
        print('MSE2:          {:}'.format(mse))
        print('RMSE2:         {:}'.format(rmse))
        print('R2:            {:}'.format(r))
        print('Time2:         {:}s'.format(time.time() - start))
        print('*' * 100)
        return None
train(args, model, optimizer)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import random
import numpy as np
import pandas as pd
import argparse
import time
from scipy.stats import pearsonr
parser = argparse.ArgumentParser()
parser.add_argument('--EPOCH', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--LR', type=float, default=0.00002, help='Initial learning rate.')
parser.add_argument('--validation_step', type=int, default=1, help='Check model for every X time')
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
def gen_gaussian_noise(signal, SNR):
    noise = np.random.randn(*signal.shape)
    noise = noise - np.mean(noise)
    signal_power = (1/signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power/np.power(10, (SNR/10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    return noise
#np.random.randint(30, 51)
for i in range(8640):
    features[i] += gen_gaussian_noise(features[i], 50)
for i in range(8640):
    for j in range(6):
        ssp = random.random()
        ssr = float(1/6)
        if ssp < ssr:
            features[i][j] = 0
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
idx_train = features[0:6912, :]
labels1_train = labels1[0:6912, :]
idx_test = features[6912:8640, :]
labels1_test = labels1[6912:8640, :]
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
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, dropout, bias):
        super(LinearRegression, self).__init__()
        self.dropout = dropout
        self.W = Parameter(torch.FloatTensor(input_size, output_size))
        self.linear = nn.Linear(input_size, output_size)
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.linear(x)
        if self.bias is not None:
            return out + self.bias
        else:
            return out
input_size = features.shape[2]
output_size = labels1.shape[2]
model = LinearRegression(input_size, output_size, dropout=0.5, bias=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
loss_func = nn.MSELoss()
def train(args, model, optimizer):
    print('Train...')
    for epoch in range(args.EPOCH):
        model.train()
        for step, (x, y) in enumerate(zip(idx_train, labels1_train)):
            x, y = torch.FloatTensor(x), torch.FloatTensor(y)
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            loss_1 = 0
            for i in range(6):
                loss = loss_func(outputs[i][0], y[i][0])
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
        for x, y in zip(idx_test, labels1_test):
            x, y = torch.FloatTensor(x), torch.FloatTensor(y)
            x, y = x.cuda(), y.cuda()
            predict = model(x)
            label = y.squeeze()
            predict = predict.squeeze()
            mae = MAE(predict, label)
            mse = MSE(predict, label)
            MAE_all.append(mae)
            MSE_all.append(mse)
            r = R(predict, label)
            R_all.append(r)
        mae = np.mean(MAE_all)
        mse = np.mean(MSE_all)
        rmse = mse ** (0.5)
        r = np.mean(R_all)
        str_ = ("%15.5g;" * 6) % (epoch + 1, mae, mse, rmse, r, time.time() - start)
        with open('LR_label1_50.csv', 'a') as f:
            f.write(str_ + '\n')
        print('EPOCH:        {:}'.format(epoch + 1))
        print('MAE:          {:}'.format(mae))
        print('MSE:          {:}'.format(mse))
        print('RMSE:         {:}'.format(rmse))
        print('R:            {:}'.format(r))
        print('Time:         {:}s'.format(time.time() - start))
        print('*' * 100)
train(args, model, optimizer)

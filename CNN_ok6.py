import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import argparse
import time
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
labels2_train = labels2[0:6912, :]
idx_test2 = features[6912:7776, :]
labels2_test2 = labels2[6912:7776, :]
idx_test = features[7776:8640, :]
labels2_test = labels2[7776:8640, :]
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
class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(CNN, self).__init__()
        self.dropout = dropout
        self.conv1 = Conv1dLayer(in_channels, 256, kernel_size=kernel_size, padding=1)
        self.conv2 = Conv1dLayer(256, 512, kernel_size=kernel_size, padding=1)
        self.conv3 = Conv1dLayer(512, out_channels, kernel_size=kernel_size, padding=1)
        self.fc1_1 = nn.Linear(out_channels * args.edge, 256)
        self.fc2_1 = nn.Linear(256, 128)
        self.fc3_1 = nn.Linear(128, 1)
        self.fc1_2 = nn.Linear(out_channels * args.edge, 256)
        self.fc2_2 = nn.Linear(256, 128)
        self.fc3_2 = nn.Linear(128, 1)
        self.fc1_3 = nn.Linear(out_channels * args.edge, 256)
        self.fc2_3 = nn.Linear(256, 128)
        self.fc3_3 = nn.Linear(128, 1)
        self.fc1_4 = nn.Linear(out_channels * args.edge, 256)
        self.fc2_4 = nn.Linear(256, 128)
        self.fc3_4 = nn.Linear(128, 1)
        self.fc1_5 = nn.Linear(out_channels * args.edge, 256)
        self.fc2_5 = nn.Linear(256, 128)
        self.fc3_5 = nn.Linear(128, 1)
        self.fc1_6 = nn.Linear(out_channels * args.edge, 256)
        self.fc2_6 = nn.Linear(256, 128)
        self.fc3_6 = nn.Linear(128, 1)
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x)
        x = x.T
        x = x.flatten()
        x11 = F.relu(self.fc1_1(x))
        x11 = F.relu(self.fc2_1(x11))
        x11 = self.fc3_1(x11)
        x21 = F.relu(self.fc1_2(x))
        x21 = F.relu(self.fc2_2(x21))
        x21 = self.fc3_2(x21)
        x31 = F.relu(self.fc1_3(x))
        x31 = F.relu(self.fc2_3(x31))
        x31 = self.fc3_3(x31)
        x41 = F.relu(self.fc1_4(x))
        x41 = F.relu(self.fc2_4(x41))
        x41 = self.fc3_4(x41)
        x51 = F.relu(self.fc1_5(x))
        x51 = F.relu(self.fc2_5(x51))
        x51 = self.fc3_5(x51)
        x61 = F.relu(self.fc1_6(x))
        x61 = F.relu(self.fc2_6(x61))
        x61 = self.fc3_6(x61)
        out = torch.cat([x11, x21, x31, x41, x51, x61], dim=0).view(6, -1)
        return out
def data_iter(features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples):
        yield features[i], labels[i]
train_loader = data_iter(idx_train, labels2_train)
test_loader = data_iter(idx_test, labels2_test)
model = CNN(in_channels=7, out_channels=64, kernel_size=3, dropout=0.5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
loss_func = nn.MSELoss()
def train(args, model, optimizer):
    print('Train...')
    for epoch in range(args.EPOCH):
        model.train()
        train_loader = data_iter(idx_train, labels2_train)
        for step, (x, y) in enumerate(train_loader):
            x, y = x.T.cuda(), y.cuda()
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
        test_loader2 = data_iter(idx_test2, labels2_test2)
        for step, (x, y) in enumerate(test_loader2):
            x, y = x.T.cuda(), y.cuda()
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
        with open('CNN_label2_no.csv', 'a') as f:
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
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data = pd.read_excel(io = r'/home/llc/Projects/bishe/totalFeatures.xls')
nrow = len(data)
ncol = len(data.columns)
row_list = []
col_list = []
for i in range(nrow):
    row = data.iloc[i, :ncol - 1]
    col = data.iloc[i, ncol - 1:]
    row_list.append(row.values)
    col_list.append(col.values[0])
x = np.array(row_list)
y = np.array(col_list)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=14)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_tensor_train = torch.from_numpy(x_train).to(torch.float32)
y_tensor_train = torch.from_numpy(y_train).to(torch.long)
x_tensor_test = torch.from_numpy(x_test).to(torch.float32)
y_tensor_test = torch.from_numpy(y_test).to(torch.long)
train_loss = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(79, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    out = net(x_tensor_train)
    loss = loss_func(out, y_tensor_train)
    train_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

out = net(x_tensor_test)
print(torch.max(out, 1)[1])
print(y_tensor_test)
print(train_loss)
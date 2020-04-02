#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.tensor as tensor # Variable

##module
from module.param import *
from module import Class

#モデルの定義
LSTM_MODEL = Class.LSTM(in_size, Node, out_size)
#オプティマイザーの定義
optimizer = optim.Adam(LSTM_MODEL.parameters())
#ロス関数の定義
LossFunction = nn.MSELoss()

# データの取得
def get_dataset():
    x = np.linspace(0, 2*np.pi, N)
    t = np.sin(x)

    x = np.array(x,dtype="float32").reshape(N,BATCH_SIZE)
    t = np.array(t,dtype="float32").reshape(N,BATCH_SIZE)
    return x, t

def train():
    print("/**********************\n*        Train        *\n**********************/")
    train_x, train_t = get_dataset()
    # 学習する関数の表示
    plt.plot(train_x, train_t, label="training data", marker=".")
    plt.show()
    loss_graph = []
    # 学習
    for epoch in range(EPOCH_NUM):
        # 誤差の初期化
        loss = 0
        # 勾配とメモリの初期化
        hidden = LSTM_MODEL.initHidden()
        LSTM_MODEL.zero_grad()

        for i in range(N-1):
            X = train_t[i].reshape(1,1)
            T = train_t[i+1].reshape(1,1)
            X = torch.from_numpy(X)
            T = torch.from_numpy(T)
            Y, hidden = LSTM_MODEL(x=X, hiddens=hidden)
            loss += LossFunction(Y, T)
        loss = loss / N
        loss.backward()
        optimizer.step()
        loss_graph.append(loss)
        # SHOW PREDICTION TIME EVERY 10 EPOCH  
        if (epoch+1) % 10 == 0:
            print("epoch:\t{}\tloss:\t{}".format(epoch+1, loss))

def test():
    test_x, test_y = get_dataset()
    LSTM_MODEL.zero_grad()
    predict_all = []
    #推論
    for i in range(N-1):
        if i == 0:
            testdata = test_y[i].reshape(1,1)
            predict_all.append(testdata)
            testdata = torch.from_numpy(testdata)
            hidden = LSTM_MODEL.initHidden()
        else:
            testdata = predict.reshape(1,1)
        predict, hidden = LSTM_MODEL(testdata, hiddens=hidden)
        predict_all.append(predict.to('cpu').detach().numpy().copy())
    # これ以降はグラフ表示のためのいろいろ
    predict_all = np.array(predict_all,dtype="float32").reshape(N)
    plt.scatter(test_x, predict_all, marker=".",label="predicted", color="red")
    plt.plot(test_x, test_y, label="true")
    plt.show()

if __name__ == "__main__":
    train()
    test()



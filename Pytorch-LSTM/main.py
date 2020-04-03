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

import os
import shutil
##module
from module.param import *
from module import Class
Class = Class

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
    train_x = train_x[:,:,np.newaxis]
    train_t = train_t[:,:,np.newaxis]
    loss_graph = []
    # 学習
    for epoch in range(EPOCH_NUM):
        # 誤差の初期化
        loss = 0
        # 勾配とメモリの初期化
        LSTM_MODEL.zero_grad()

        X = train_x
        T = train_t
        X = torch.from_numpy(X)
        T = torch.from_numpy(T)
        Y = LSTM_MODEL(x=X)
        loss = LossFunction(Y, T)

        loss.backward()
        optimizer.step()
        loss_graph.append(loss)
        # SHOW PREDICTION TIME EVERY 10 EPOCH  
        if (epoch+1) % 10 == 0:
            print("epoch:\t{}\tloss:\t{}".format(epoch+1, loss))

def test():
    test_x, test_t = get_dataset()
    test_x = test_x[:,:,np.newaxis]
    test_t = test_t[:,:,np.newaxis]

    LSTM_MODEL.zero_grad()
    predict_all = []
    #推論

    testdata = test_x
    testdata = torch.from_numpy(testdata)
    predict = LSTM_MODEL(testdata)
    predict = predict.to('cpu').detach().numpy().copy()
    # これ以降はグラフ表示のためのいろいろ
    plt.scatter(test_x.reshape(N), predict.reshape(N), marker=".",label="predicted", color="red")
    plt.plot(test_x.reshape(N), test_t.reshape(N), label="true")
    plt.show()

if __name__ == "__main__":
    train()
    test()



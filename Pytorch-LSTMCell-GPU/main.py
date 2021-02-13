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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#モデルの定義
LSTM_MODEL = Class.LSTM(in_size=in_size, hidden_size=Node, out_size=out_size, hidden_lstm_layer=hidden_lstm_layer)
LSTM_MODEL = LSTM_MODEL.to(device)
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
    # 学習
    for epoch in range(EPOCH_NUM):
        # 誤差の初期化
        loss = 0
        # 勾配とメモリの初期化
        LSTM_MODEL.zero_grad()
        hidden = LSTM_MODEL.initHidden()

        for i in range(N-1):
            X = train_t[i].reshape(BATCH_SIZE,in_size)
            T = train_t[i+1].reshape(BATCH_SIZE,out_size)
            X = torch.from_numpy(X)
            T = torch.from_numpy(T)
            X = X.to(device)
            T = T.to(device)
            Y, hidden = LSTM_MODEL.forward(x=X, hiddens=hidden)
            loss += LossFunction(Y, T)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print("epoch:\t{}\tloss:\t{:.5f}".format(epoch+1, loss))
            if (epoch+1) == 500:
                torch.save(LSTM_MODEL.state_dict(), "model/LSTM_EPOCH{}.npz".format(epoch+1))


def test():
    test_x, test_y = get_dataset()
    LSTM_MODEL.zero_grad()
    predict_all = []
    #推論
    for i in range(N-1):
        if i == 0:
            testdata = test_y[i].reshape(BATCH_SIZE,in_size)
            predict_all.append(testdata)
            testdata = torch.from_numpy(testdata)
            testdata = testdata.to(device)
            hidden = LSTM_MODEL.initHidden()
        else:
            testdata = predict.reshape(BATCH_SIZE,out_size)
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



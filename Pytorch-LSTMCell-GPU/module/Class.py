#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.tensor as tensor
from module.param import *
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# モデルクラス定義
class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, hidden_lstm_layer):
        super(LSTM, self).__init__()
        self.hidden_layer_list = []
        self.hidden_size = hidden_size
        self.hidden_lstm_layer = hidden_lstm_layer
        self.input_layer = nn.LSTMCell(in_size, hidden_size)
        
        for i in range(self.hidden_lstm_layer):
            name = "lstm{}".format(i)
            exec('self.{} = nn.LSTMCell(hidden_size, hidden_size)'.format(name))
            exec('self.{} = self.{}.to(device)'.format(name,name))
            exec('self.hidden_layer_list.append(self.{})'.format(name))
        self.output_layer = nn.Linear(hidden_size, out_size)
    def forward(self, x, hiddens):
        hiddens_return = []
        h, c = self.input_layer(x,hiddens[0])
        hiddens_return.append([h,c])
        for hid_lstm in range(self.hidden_lstm_layer):
            model = self.hidden_layer_list[hid_lstm]
            h, c = model(h,hiddens[hid_lstm+1])
            hiddens_return.append([h,c])
        output = self.output_layer(h)

        return output, hiddens_return

    def initHidden(self):
        hidden_init = []
        for j in range(self.hidden_lstm_layer+1):#入力レイヤー分も加える+1
            hid = []
            for k in range(2):
                h = torch.zeros(BATCH_SIZE, self.hidden_size)
                h = h.to(device)
                hid.append(h)
            hidden_init.append(hid)
        return hidden_init
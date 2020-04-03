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
# モデルクラス定義
class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(in_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x, hiddens):       
        h1, c1 = self.lstm1(x,hiddens[0])
        h2, c2 = self.lstm2(h1,hiddens[1])
        output = self.linear(h2)
        return output, [ [h1, c1], [h2, c2] ]

    def initHidden(self):
        # LSTM2層の場合は[[h1,c1], [h1, c1]]で初期化
        return [ [torch.zeros(BATCH_SIZE, self.hidden_size), torch.zeros(BATCH_SIZE, self.hidden_size)],[torch.zeros(BATCH_SIZE, self.hidden_size), torch.zeros(BATCH_SIZE, self.hidden_size)]  ]
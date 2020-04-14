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
        self.num_layers = 2
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):       
        # LSTM2層の場合は[[h1,c1], [h1, c1]]で初期化
        #ここ自動化しないと
        hiddens = [ torch.zeros(self.num_layers, 1, self.hidden_size), torch.zeros(self.num_layers, 1, self.hidden_size)]
        output, [h,c] = self.lstm(x,hiddens)
        output = self.linear(output)
        return output
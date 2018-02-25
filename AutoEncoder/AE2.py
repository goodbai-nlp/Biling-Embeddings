#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: BAE.py
@time: 18-1-12 下午4:55
"""
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataset import MyDataSet
import embeddings
from logger import Logger


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encode_0 = nn.Linear(640,512)
        self.encode_1 = nn.Linear(512,128,False)
        # self.encode_2 = nn.Linear(128, 100,False)

        #self.decode_0 = nn.Linear(100, 128,False)
        self.decode_1 = nn.Linear(128,512,False)
        self.decode_2 = nn.Linear(512,640)

        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()



    def encode(self, x):
        encoded = self.relu(self.encode_0(x))
        # encoded = self.relu(self.encode_1(encoded))
        return self.encode_1(encoded)

    def decode(self, z):
        decoded = self.relu(self.decode_1(z))
        # decoded = self.relu(self.decode_1(decoded))
        return (self.decode_2(decoded))

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded


class BasicLinear(nn.Module):
    def __init__(self, layers_size):
        super(BasicLinear, self).__init__()
        self.linear_layers = []
        for i in range(len(layers_size) - 1):
            self.__setattr__("linear_layer_{}".format(i), nn.Linear(layers_size[i], layers_size[i + 1]))

        self.layers_count = len(layers_size) - 1

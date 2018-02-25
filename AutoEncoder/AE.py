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
    def __init__(self, layers_size):
        super(AE, self).__init__()
        self.linear_layers = []

        '''encoder'''
        en_layers_size = layers_size[0]
        for i in range(len(en_layers_size) - 1):
            self.__setattr__("encode_layer_{}".format(i), nn.Linear(en_layers_size[i], en_layers_size[i + 1]))

        self.encode_layers_count = len(en_layers_size) - 1

        '''decoder'''
        de_layers_size = layers_size[1]
        for i in range(len(de_layers_size) - 1):
            self.__setattr__("decode_layer_{}".format(i), nn.Linear(de_layers_size[i], de_layers_size[i + 1]))
        self.decode_layers_count = len(de_layers_size) - 1

        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()

    def encode(self, x):
        encoded = x
        for i in range(self.encode_layers_count):
            encoded = self.__getattr__("encode_layer_{}".format(i))(encoded)
            encoded = self.relu(encoded)
        return encoded

    def decode(self, z):
        out =z
        for i in range(self.decode_layers_count - 1):
            out = self.__getattr__("decode_layer_{}".format(i))(out)
            out = self.relu(out)
        out = self.sigmod(self.__getattr__("decode_layer_{}".format(self.decode_layers_count - 1))(out))

        return out

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

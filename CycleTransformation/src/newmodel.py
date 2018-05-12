#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: model.py
@time: 18-3-30 下午4:23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


class GaussianNoiseLayer(nn.Module):
    def __init__(self, sigma,shape):
        super(GaussianNoiseLayer,self).__init__()
        self.sigma = sigma
        self.noise = Variable(torch.zeros(128,shape).cuda())

    def forward(self, x):
        self.noise.data.normal_(1, std=self.sigma)
        return x*self.noise

class Discriminator(nn.Module):

    def __init__(self, params,dis_layers):
        super(Discriminator, self).__init__()
        self.emb_dim = params.d_input_size
        self.dis_layers = dis_layers

        self.dis_hidden_dropout = params.dis_hidden_dropout
        self.dis_input_dropout = params.dis_input_dropout
        self.activation = params.dis_activation

        layers = [nn.Dropout(self.dis_input_dropout)]

        for i in range(len(self.dis_layers)-1):
            input_dim,output_dim = self.dis_layers[i],self.dis_layers[i+1]
            layers.append(nn.Linear(input_dim,output_dim))
            if i< len(self.dis_layers)-2:
                if self.activation =='leakyrelu':
                    layers.append(nn.LeakyReLU(0.2))        # 激活可调
                elif self.activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
                # layers.append(nn.LeakyReLU(0.2))        # 激活可调
                layers.append(nn.Dropout(self.dis_hidden_dropout))
        layers.append(nn.Sigmoid())
        self.all_layer = layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() ==2 and x.size(1) == self.emb_dim

        return self.layers(x).view(-1)

class Generator(nn.Module):

    def __init__(self, params,gen_layers,lineartag):
        super(Generator, self).__init__()
        self.emb_dim = params.g_input_size
        self.gen_layers = gen_layers
        self.activation = params.gen_activation
        self.Matrix = lineartag
        self.all_layer = []
        layers = []

        for i in range(len(self.gen_layers)-1):
            input_dim,output_dim = self.gen_layers[i],self.gen_layers[i+1]
            if not self.Matrix:
                layers.append(nn.Linear(input_dim,output_dim))
            else:
                layers.append(nn.Linear(input_dim,output_dim,bias=False))

            if i< len(self.gen_layers)-2:
                if self.activation =='leakyrelu':
                    layers.append(nn.LeakyReLU(0.2))        # 激活可调
                elif self.activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
                # layers.append(nn.Dropout(self.dis_hidden_dropout))
        if not self.Matrix:
            layers.append(nn.Tanh())

        self.all_layer = layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x)

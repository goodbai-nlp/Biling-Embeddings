#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: BAE.py
@time: 18-1-16 下午3:30
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
import sys

torch.cuda.set_device(1)
from AE2 import AE


class MyLossFunc(nn.Module):

    def __init__(self, alpha, beta):
        super(MyLossFunc, self).__init__()
        self.alpha = alpha
        self.beta = beta
        return

    def pca(self, x, n_components):
        m = x.size()[0]
        mean1 = torch.mean(x, dim=0)
        H1bar = x - mean1.repeat(m, 1)
        cov_Mat = (1.0 / (m - 1)) * torch.mm(H1bar.t(), H1bar)
        u, s, v = torch.svd(cov_Mat)

        trans = torch.mm(H1bar, u[:, :n_components])
        return trans

    def forward(self, view1, view2, x1, x2, t1, t2, w1, w2):
        '''Loss forward function'''

        mse1 = nn.MSELoss(size_average=False)
        mse2 = nn.MSELoss(size_average=False)

        '''Cosine Loss'''
        # t1 = self.pca(t1,100)
        # t2 = self.pca(t2,100)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_loss = torch.sum(cos(t1, t2))

        '''F-loss'''
        diff = t1 - t2
        out = torch.pow(diff, 2).sum(dim=1, keepdim=True)
        F_loss = torch.pow(out.sum(), 0.5)

        '''param regular'''
        tmp = w1.data.size()[1]
        R1 = torch.dist(torch.mm(w1.t(), w1), Variable(torch.eye(tmp).cuda()), 2)
        R2 = torch.dist(torch.mm(w2.t(), w2), Variable(torch.eye(tmp).cuda()), 2)
        R = self.alpha * (R1 + R2)

        '''Reconstruction Loss'''
        view1_rec_loss = mse1(x1, view1)
        view2_rec_loss = mse2(x2, view2)
        reconstruction_loss = view1_rec_loss + view2_rec_loss
        # reconstruction_loss = view1_rec_loss
        # return -cos_loss + R + self.beta * reconstruction_loss
        return (-cos_loss, R, self.beta * reconstruction_loss)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # self.view1_AE = AE([[640, 512], [512, 640]])
        # self.view1_enfc = nn.Linear(512, 128, bias=False)  # view1 正交层
        # self.view1_defc = nn.Linear(128, 512, bias=False)
        #
        # self.view2_AE = AE([[640, 512], [512, 640]])
        # self.view2_enfc = nn.Linear(512, 128, bias=False)  # view2 正交层
        # self.view2_defc = nn.Linear(128, 512, bias=False)
        # self.relu = nn.ReLU()

        self.view1_AE = AE()
        self.view2_AE = AE()

    def forward(self, view1, view2):
        # z1 = self.view1_AE.encode(view1)
        # z1 = self.view1_enfc(z1)
        #
        # x1 = self.view1_defc(z1)
        # x1 = self.relu(x1)
        # x1 = self.view1_AE.decode(x1)
        #
        # z2 = self.view2_AE.encode(view2)
        # z2 = self.view2_enfc(z2)
        #
        # x2 = self.view2_defc(z2)
        # x2 = self.relu(x2)
        # x2 = self.view2_AE.decode(x2)
        z1,x1 = self.view1_AE(view1)
        z2,x2 = self.view2_AE(view2)

        return (z1, x1, z2, x2)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.1)


if __name__ == "__main__":

    epoch_num = 800
    batch_size = 3000
    learn_rate = 0.001
    # batch_size = int(sys.argv[1])
    alpha = 100
    beta = 1000
    # alpha = int(sys.argv[1])

    logger = Logger('./logs')

    net = Net()
    net.cuda()
    criterion = MyLossFunc(alpha, beta)

    optimizer1 = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=True)
    optimizer2 = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9, weight_decay=True)
    optimizer3 = optim.Adam(net.parameters(), lr=learn_rate, betas=(0.9, 0.99))
    optimizer4 = optim.LBFGS(net.parameters(), lr=0.8)

    trainSet = MyDataSet(root='en.train640 de.train640', train=True, transform=None)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    net.apply(weights_init)
    step = 0
    l1,l2,l3 = 0,0,0
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            step += 1
            rowLen = int(data.size()[1] / 2)
            input_view1, input_view2 = data[:, :rowLen], data[:, rowLen:]

            input_view1, input_view2 = Variable(input_view1), Variable(input_view2)  # 把数据放到gpu中

            optimizer3.zero_grad()

            z1, x1, z2, x2 = net(input_view1.cuda().float(), input_view2.cuda().float())

            l1,l2,l3 = criterion(input_view1.cuda().float(), input_view2.cuda().float(), x1, x2, z1, z2,
                                   net.view1_AE.encode_1.weight, net.view2_AE.encode_1.weight)

            loss = l1+l2+l3
            loss.backward()

            optimizer3.step()

            # print statistics
            running_loss += loss.data[0]

            info = {
                'BiAE_Loss3': loss.data[0],
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5),end='\n')
                #print('Cos loss: {}, reconstruction_loss:{}'.format(l1, l3))
                print('Cos loss: {}, regluar_loss:{}, reconstruction_loss:{}'.format(l1,l2,l3))
                running_loss = 0.0
                # print(net.view1_fc.weight.grad)

    # print(net.view1_AE.encode_layer_0.weight.data)
    print('Cos loss: {}, regluar_loss:{}, reconstruction_loss:{}'.format(l1, l2, l3))

    source_file = open('new_embedding_size640.en', encoding='utf-8', errors='surrogateescape')
    target_file = open('new_embedding_size640.de', encoding='utf-8', errors='surrogateescape')
    en_words, en_vec = embeddings.read(source_file)
    de_words, de_vec = embeddings.read(target_file)

    en_vec = embeddings.length_normalize(en_vec)
    de_vec = embeddings.length_normalize(de_vec)

    input_view1, input_view2 = Variable(torch.from_numpy(en_vec).cuda()), Variable(torch.from_numpy(de_vec).cuda())

    res_envec, x1, res_devec, x2 = net(input_view1.float(), input_view2.float())
    print(x1)

    src_file = open('BiAE.en', mode='w', encoding='utf-8', errors='surrogateescape')
    trg_file = open('BiAE.de', mode='w', encoding='utf-8', errors='surrogateescape')

    # res_envec = embeddings.length_normalize(res_envec.data.cpu().numpy())
    # res_devec = embeddings.length_normalize(res_devec.data.cpu().numpy())

    res_envec = (res_envec.data.cpu().numpy())
    res_devec = (res_devec.data.cpu().numpy())

    embeddings.write(en_words, res_envec, src_file)
    embeddings.write(de_words, res_devec, trg_file)

    source_file.close()
    target_file.close()
    src_file.close()
    trg_file.close()
    print('Finished Training')

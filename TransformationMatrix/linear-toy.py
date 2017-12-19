#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: linear-toy.py
@time: 17-12-15 上午10:47
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from dataset import MyDataSet
import embeddings

torch.cuda.set_device(2)


class MyLossFunc(nn.Module):
    def __init__(self, alpha):
        super(MyLossFunc, self).__init__()
        self.alpha = alpha

        return

    def forward(self, t1,t2,w1,w2):

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        cos_sim = torch.sum(cos(t1,t2))
        # print(w1.size())
        w1_loss = w1.mm(torch.t(w1)) - torch.eye(w1.size()[0]).cuda()  #

        w1_Floss = np.linalg.norm(w1_loss.cpu().numpy(),'fro')

        w2_loss = w2.mm(torch.t(w2)) - torch.eye(w2.size()[0]).cuda()
        w2_Floss = np.linalg.norm(w2_loss.cpu().numpy(), 'fro')

        weight_loss = self.alpha*(w1_Floss+w2_Floss)

        return -cos_sim + weight_loss

        #return -cos_sim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.view1_fc = nn.Linear(200,100,bias=False)
        self.view2_fc = nn.Linear(200,100,bias=False)


    def forward(self, *input):
        x1,x2 = input[0],input[1]
        out1 = self.view1_fc(x1)
        out2 = self.view2_fc(x2)

        return out1,out2



if __name__ == "__main__":

    alpha = 0.1


    net = Net()
    net.cuda()
    criterion = MyLossFunc(alpha)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainSet =  MyDataSet(root='en.train de.train', train=True,transform=None)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=250,
                                          shuffle=True, num_workers=2)


    # testSet = MyDataSet(root='new_embedding_size200.en new_embedding_size200.de', train=True, transform=None)
    # testloader = torch.utils.data.DataLoader(testSet, batch_size=3000,
    #                                              shuffle=True, num_workers=2)

    nn.init.orthogonal(net.view1_fc.weight.data)
    nn.init.orthogonal(net.view2_fc.weight.data)


    for epoch in range(300):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            rowLen = int(data.size()[1]/2)
            input_view1,input_view2 = data[:,:rowLen], data[:,rowLen:]

            input_view1,input_view2 = Variable(input_view1.cuda()),Variable(input_view2.cuda()) #把数据放到gpu中
            optimizer.zero_grad()

            #forward+backward+optimize
            outputs = net(input_view1.float(),input_view2.float())          #试试floattensor
            loss = criterion(outputs[0],outputs[1],net.view1_fc.weight.data,net.view2_fc.weight.data)
            loss.backward()
            optimizer.step()

            #print statistics
            running_loss+=loss.data[0]
            if i%20==19:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
                running_loss=0.0

    source_file = open('new_embedding_size200.en', mode='w', encoding='utf-8', errors='surrogateescape')
    target_file = open('new_embedding_size200.de', mode='w',encoding='utf-8', errors='surrogateescape')
    en_words, en_vec = embeddings.read(source_file)
    de_words, de_vec = embeddings.read(target_file)

    input_view1, input_view2 = Variable(torch.from_numpy(en_vec).cuda()), Variable(torch.from_numpy(de_vec).cuda())

    res_envec,res_devec = net(input_view1.float(),input_view2.float())

    src_file = open('res.en', encoding='utf-8', errors='surrogateescape')
    trg_file = open('res.de', encoding='utf-8', errors='surrogateescape')
    embeddings.write(en_words, res_envec.numpy(), src_file)
    embeddings.write(de_words, res_devec.numpy(), trg_file)

    source_file.close()
    target_file.close()
    src_file.close()
    trg_file.close()
    print('Finished Training')

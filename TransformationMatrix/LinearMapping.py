#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: LinearMapping.py
@time: 17-12-20 下午3:13
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import MyDataSet
import embeddings
from logger import Logger
import sys

torch.cuda.set_device(2)

class MyLossFunc(nn.Module):
    def __init__(self, alpha):
        super(MyLossFunc, self).__init__()
        self.alpha = alpha
        return

    def forward(self, t1,t2, w1):

        # cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        # cos_sim = torch.sum(cos(t1,t2))
        diff = t1-t2
        out = torch.pow(diff , 2).sum(dim=1, keepdim=True)
        #F_loss = torch.pow(out.sum(),0.5)
        F_loss = torch.pow(out.sum(),0.5)
        #mse_loss = np.linalg.norm(diff.data.cpu().numpy(),'fro')
        # w1_loss = w1.mm(torch.t(w1)) - torch.eye(w1.size()[0]).cuda()  #
        # w1_Floss = np.linalg.norm(w1_loss.cpu().numpy(),'fro')
        # weight_loss = self.alpha*(w1_Floss)
        #return -cos_sim + weight_loss
        #return mse_loss + weight_loss
        return F_loss

        #return Variable(torch.FloatTensor([float(mse_loss)]), requires_grad=True)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.view1_fc = nn.Linear(200,200,bias=False)

    def forward(self, input):
        x1 = input
        out1 = self.view1_fc(x1)
        return out1


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        #m.weight.data = 1.0*torch.ones(m.weight.size()).cuda()
        # nn.init.constant(m.bias,0.1)

if __name__ == "__main__":

    batch_size = 3000
    #batch_size = int(sys.argv[1])
    alpha = 1
    logger = Logger('./logs')

    net = Net()
    net.cuda()

    # loss = nn.MSELoss()

    # criterion = MyLossFunc(alpha)
    criterion2 = nn.MSELoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=True)
    #optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9,weight_decay=True)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))
    # optimizer = optim.LBFGS(net.parameters(),lr=0.8)

    trainSet =  MyDataSet(root='en.train de.train', train=True,transform=None)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    net.apply(weights_init)
    step = 0
    for epoch in range(600):

        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            step += 1
            rowLen = int(data.size()[1]/2)
            input_view1,input_view2 = data[:,:rowLen], data[:,rowLen:]

            input_view1, input_view2 = Variable(input_view1,requires_grad=True), Variable(input_view2) #把数据放到gpu中

            optimizer.zero_grad()

            #forward+backward+optimize
            outputs = net(input_view1.float().cuda())          #GPU中是floattensor 不是doubletensor

            #loss = criterion2(outputs,input_view2.float(),net.view1_fc.weight.data)
            loss = criterion2(outputs,input_view2.float().cuda())
            # def closure():
            #     optimizer.zero_grad()
            #     outp = net(input_view1.float().cuda())
            #     #loss = criterion(outp, input_view2.cuda().float(),net.view1_fc.weight.data)
            #     loss = criterion2(outp,input_view2.float().cuda())
            #     loss.backward()
            #
            #     info = {
            #         'loss': loss.data[0],
            #     }
            #     for tag, value in info.items():
            #         logger.scalar_summary(tag, value, step)
            #     return loss

            loss.backward()
            optimizer.step()

            #print statistics
            running_loss+=loss.data[0]

            info = {
                'loss': loss.data[0],
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            if i%5==4:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 5))
                running_loss=0.0
                # print(net.view1_fc.weight.grad)

    source_file = open('new_embedding_size200.en', encoding='utf-8', errors='surrogateescape')
    target_file = open('new_embedding_size200.de', encoding='utf-8', errors='surrogateescape')
    en_words, en_vec = embeddings.read(source_file)
    de_words, de_vec = embeddings.read(target_file)

    en_vec = embeddings.length_normalize(en_vec)
    de_vec = embeddings.length_normalize(de_vec)

    input_view1, input_view2 = Variable(torch.from_numpy(en_vec).cuda()), Variable(torch.from_numpy(de_vec).cuda())

    res_envec= net(input_view1.float())

    src_file = open('LinearMappingres.en', mode='w',encoding='utf-8', errors='surrogateescape')
    trg_file = open('LinearMappingres.de', mode='w',encoding='utf-8', errors='surrogateescape')

    res_envec = embeddings.length_normalize(res_envec.data.cpu().numpy())

    embeddings.write(en_words, res_envec, src_file)
    embeddings.write(de_words, input_view2.float().data.cpu().numpy(), trg_file)

    source_file.close()
    target_file.close()
    src_file.close()
    trg_file.close()
    print('Finished Training')
    # print(net.view1_fc.weight.data)

#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: MyGAN.py
@time: 18-1-27 上午11:10
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from embedding import WordEmbeddings
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
import sys
import time
from AE import AE

torch.cuda.set_device(2)

g_input_size = 50  # Random noise dimension coming into generator, per output vector

g_hidden_size1 = 100  # Generator complexity
g_hidden_size2 = 100  # Generator complexity
g_output_size = 50  # size of generated output vector

d_input_size = 50  # Minibatch size - cardinality of distributions
d_hidden_size = 500  # Discriminator complexity
d_hidden_size2 = 200  # Discriminator complexity
d_output_size = 1  # Single dimension for 'real' vs. 'fake'

HALF_BATCH_SIZE = 128

# d_learning_rate = 2e-4  # 2e-4
d_learning_rate = 0.001
# g_learning_rate = 2e-4
g_learning_rate = 0.001

optim_betas = (0.9, 0.999)
num_epochs = 50000
print_interval = 100
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

input_noise = 0.5
hidden_noise = 0.5

gloss_min = 100000

recon_weight = 1

TrainNew = int(sys.argv[1])
from logger import Logger

class GaussianNoiseLayer(nn.Module):
    def __init__(self, sigma,shape):
        super(GaussianNoiseLayer,self).__init__()
        self.sigma = sigma
        self.noise = Variable(torch.zeros(HALF_BATCH_SIZE,shape).cuda())

    def forward(self, x):
        self.noise.data.normal_(1, std=self.sigma)
        # GaussianNoise = Variable(torch.ones(x.size()).normal_(1.0,self.sigma)).cuda().float()
        # GaussianNoise = Variable(torch.normal(ones,std=self.sigma)).cuda().float()
        return x*self.noise



class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.map = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.map(x)
        recon = F.linear(x, self.map.weight.t(), bias=None)
        return x, recon

# class Generator1(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Generator1, self).__init__()
#         self.map = nn.Linear(input_size, output_size)
#         self.map2 = nn.Linear(output_size,input_size)
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         x = self.tanh(self.map(x))
#         # recon = F.linear(x, self.map.weight.t(), bias=None)
#         recon = self.map2(x)
#         recon = self.tanh(recon)
#
#         return x, recon
#
#
# class Generator2(nn.Module):
#     def __init__(self):
#         super(Generator2, self).__init__()
#         self.AE = AE([[g_input_size, g_hidden_size1], [g_hidden_size1, g_output_size]])
#         self.view1_enfc = nn.Linear(g_hidden_size1, g_output_size, bias=False)  # view1 正交层
#         self.view1_defc = nn.Linear(g_output_size, g_hidden_size1, bias=False)
#         self.relu = nn.ReLU()
#         self.relu = nn.LeakyReLU()
#
#     def forward(self, x):
#         z = self.AE.encode(x)
#         z = self.view1_enfc(z)
#
#         recon = self.view1_defc(z)
#         recon = self.relu(recon)
#         recon = self.AE.decode(recon)
#
#         return z, recon


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()


        self.input_noise = GaussianNoiseLayer(sigma=0.5,shape=input_size)
        self.map1 = nn.Linear(input_size, hidden_size)
        # self.map2 = nn.Linear(hidden_size, d_hidden_size2)
        self.map2 = nn.Linear(hidden_size, d_hidden_size2)
        self.map3 = nn.Linear(d_hidden_size2, output_size)
        self.map21 = nn.Linear(d_hidden_size, output_size)

        # self.hidden_noise = GaussianNoiseLayer(sigma=0.5,shape=hidden_size)
        self.sigmod = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # dropout 加在线性变换前面
        net = F.dropout(x, 0.5, training=self.training)
        # net = self.input_noise(x)
        # print(net.size())
        net = self.map1(net)
        net = self.relu(net)
        # net = self.relu(self.map1(net))
        # net = self.hidden_noise(net)
        # net = F.dropout(net, 0.5, training=self.training)
        # net = self.relu(self.map2(net))
        # net = self.sigmod(self.map3(net))
        net = self.sigmod(self.map21(net))


        return net


def weight_init2(m):
    # 参数初始化。 可以改成xavier初始化方法
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.01)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.01)


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


G = Generator(input_size=g_input_size, output_size=g_output_size)
# G = Generator1(input_size=g_input_size, output_size=g_output_size)
# G = Generator2()

D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
criterion2 = nn.CosineSimilarity(dim=1, eps=1e-6)

d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas, weight_decay=0)

G.cuda()
D.cuda()
G.apply(weights_init)
D.apply(weight_init2)

dataDir = './'
rng = check_random_state(3)

we1 = WordEmbeddings()
we1.load_from_word2vec(dataDir, 'zh')
we1.downsample_frequent_words()
we1.vectors = normalize(we1.vectors)
we_batches1 = we1.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

we2 = WordEmbeddings()
we2.load_from_word2vec(dataDir, 'en')
we2.downsample_frequent_words()
we2.vectors = normalize(we2.vectors)
we_batches2 = we2.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

assert we1.embedding_dim == we2.embedding_dim
d = we1.embedding_dim
logger = Logger('./logs')
start_time = time.time()

if TrainNew:

    for epoch in range(num_epochs):
        id1 = next(we_batches1)
        id2 = next(we_batches2)

        input_data = Variable(torch.from_numpy(we1.vectors[id1]))
        trg_data = Variable(torch.from_numpy(we2.vectors[id2]))
        # X = torch.Tensor(HALF_BATCH_SIZE, d)


        for d_index in range(d_steps):
            for p in D.parameters():
                p.requires_grad = True  # to avoid computation
            D.train()
            # 1. Train D on real+fake
            D.zero_grad()
            #  1A: Train D on real
            d_real_decision = D(trg_data.cuda().float())
            d_real_error = criterion(d_real_decision, Variable(torch.ones(HALF_BATCH_SIZE, 1)).cuda())  # ones = true
            # d_real_error.backward() # compute/store gradients, but don't change params

            #  1B: Train D on fake

            g_fake_data, g_recon_data = G(input_data.cuda().float())  # detach to avoid training G on these labels,假设G固定
            # X = g_fake_data.data.clone()

            g_fake_data.detach()
            # g_recon_data.detach()
            # d_fake_data = G(d_input_data.cuda().float())  # detach to avoid training G on these labels,假设G固定

            # fake = Variable(X).cuda().float()
            d_fake_decision = D(g_fake_data)
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(HALF_BATCH_SIZE, 1)).cuda())  # zeros = fake

            d_error = (d_real_error + d_fake_error) / 2.0
            # d_fake_error.backward()
            d_error.backward()

            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(g_steps):
            for p in D.parameters():
                p.requires_grad = False  # to avoid computation

            G.train()
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            g_fake_data, g_recon_data = G(input_data.cuda().float())
            dg_fake_decision = D(g_fake_data)

            # g_error = criterion(dg_fake_decision, Variable(torch.ones(HALF_BATCH_SIZE,1)).cuda())  # we want to fool, so pretend it's all genuine
            g_error = - torch.mean(torch.log(dg_fake_decision))  # use -log to replace log(1-p)
            g_recon_loss = 1.0 - torch.mean(criterion2(input_data.cuda().float(), g_recon_data))
            loss = g_error + recon_weight * g_recon_loss

            loss.backward()
            g_optimizer.step()  # Only optimizes G's parameters



        if epoch % print_interval == 0:
            pass
            # print("d_loss:{:.4f} g_adv_loss:{:.4f} recon_gen_loss_val:{:.4f} ".format(d_error.data[0],g_error.data[0],g_recon_loss.data[0]))
        info = {
            'Rec_loss3': loss.data[0],
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        if (epoch > 10000) and (loss.data[0] < gloss_min):
            gloss_min = loss.data[0]
            # W = G.map.weight.data.cpu().numpy()
            torch.save(G.state_dict(), 'g_params_min.pkl')
            print("epoch:{} sum_loss:{}".format(epoch, loss.data[0]))

            # print(" recon_gen_loss_val:{}  ||W^T*W - I||:{}".format(g_recon_loss.data[0],
            #                                                         np.linalg.norm(np.dot(W.T, W) - np.identity(d))))
            print("d_loss:{:.4f} g_adv_loss:{:.4f} recon_gen_loss_val:{:.4f} ".format(d_error.data[0],g_error.data[0],g_recon_loss.data[0]))
            # we1.transformed_vectors = np.dot(we1.vectors, W.T)
            # we1.save_transformed_vectors(dataDir + '/UBiLexAT/data/zh-en/transformed-1' + '.' + 'zh')

    torch.save(G.state_dict(), 'g_params_final.pkl')

print('Training time', (time.time() - start_time) / 60, 'min')


G2 = Generator(input_size=g_input_size, output_size=g_output_size).cuda()
# G2 = Generator2().cuda()
# G2 = Generator1(input_size=g_input_size, output_size=g_output_size).cuda()

G2.load_state_dict(torch.load('g_params_min.pkl'))

d_input_data_all = Variable(torch.from_numpy(we1.vectors))

transformed_data,_ = G2(d_input_data_all.cuda().float())

we1.transformed_vectors = transformed_data.data.cpu().numpy()
# W = G2.map.weight.data.cpu().numpy()
#
# we1.transformed_vectors = np.dot(we1.vectors, W)
we1.save_transformed_vectors(dataDir + '/UBiLexAT/data/zh-en/transformed-1' + '.' + 'zh')
print('All running time',(time.time() - start_time) / 60, 'min')


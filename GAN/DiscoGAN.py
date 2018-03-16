#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: DiscoGAN.py
@time: 18-3-12 上午11:19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable

from embedding import WordEmbeddings
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from logger import Logger
import sys
import time
torch.cuda.set_device(3)

# hyper params
HALF_BATCH_SIZE = 128
optim_betas = (0.9, 0.999)
num_epochs = 50000

print_interval = 100
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

input_noise = 0.5
hidden_noise = 0.5
gloss_min = 100000

recon_weight = 1

g_input_size = 50     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 50    # size of generated output vector
d_input_size = 50   # Minibatch size - cardinality of distributions
d_hidden_size = 500   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'

d_learning_rate = 0.001
g_learning_rate = 0.001


class GaussianNoiseLayer(nn.Module):
    def __init__(self, sigma,shape):
        super(GaussianNoiseLayer,self).__init__()
        self.sigma = sigma
        self.noise = Variable(torch.zeros(HALF_BATCH_SIZE,shape).cuda())

    def forward(self, x):
        self.noise.data.normal_(1, std=self.sigma)
        return x*self.noise


class Generator(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.map = torch.nn.Linear(input_size, output_size,bias=False)

    def forward(self, x):
        x = self.map(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        self.input_noise = GaussianNoiseLayer(sigma=0.5,shape=input_size)
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)
        self.hidden_noise = GaussianNoiseLayer(sigma=0.5,shape=hidden_size)
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # dropout 加在线性变换前面
        net = F.dropout(x, 0.5, training=self.training)
        # net = self.input_noise(x)
        net = self.relu(self.map1(net))
        # net = self.hidden_noise(net)
        # net = F.dropout(net, 0.5, training=self.training)
        net = self.sigmod(self.map2(net))

        return net


def log(x):
    return torch.log(x + 1e-8)


def weight_init2(m):
    # 参数初始化。 可以改成xavier初始化方法
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            torch.nn.init.constant(m.bias, 0.01)

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal(m.weight)
        if m.bias is not None:
            torch.nn.init.constant(m.bias, 0.01)


# G_AB = torch.nn.Sequential(
#     torch.nn.Linear(g_input_size, g_output_size, bias=False)
#     # torch.nn.ReLU(),
#     # torch.nn.Linear(h_dim, X_dim),
#     # torch.nn.Sigmoid()
# )
#
# G_BA = torch.nn.Sequential(
#     torch.nn.Linear(g_input_size, g_output_size, bias=False)
#     # torch.nn.ReLU(),
#     # torch.nn.Linear(h_dim, X_dim),
#     # torch.nn.Sigmoid()
# )

G_AB = Generator(input_size=g_input_size, output_size=g_output_size)
G_BA = Generator(input_size=g_input_size, output_size=g_output_size)

D_A = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
D_B = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)


# D_A = torch.nn.Sequential(
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(d_input_size, d_hidden_size),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(d_hidden_size, 1),
#     torch.nn.Sigmoid()
# )
#
# D_B = torch.nn.Sequential(
#     torch.nn.Linear(d_input_size, d_hidden_size),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(d_hidden_size, 1),
#     torch.nn.Sigmoid()
# )


dataDir = './'
rng = check_random_state(0)

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



nets = [G_AB, G_BA, D_A, D_B]
G_params = list(G_AB.parameters()) + list(G_BA.parameters())
D_params = list(D_A.parameters()) + list(D_B.parameters())

for net in nets:
    net.cuda()

G_AB.apply(weights_init)
G_BA.apply(weights_init)
D_A.apply(weight_init2)
D_B.apply(weight_init2)

def reset_grad():
    for net in nets:
        net.zero_grad()


G_solver = optim.Adam(G_params, lr=d_learning_rate,betas=optim_betas)
D_solver = optim.Adam(D_params, lr=g_learning_rate,betas=optim_betas)



TrainNew = int(sys.argv[1])

if TrainNew:
    # Training
    for it in range(num_epochs):
        # Sample data from both domains
        id1 = next(we_batches1)
        id2 = next(we_batches2)

        X_A = Variable(torch.from_numpy(we1.vectors[id1])).cuda().float()
        X_B = Variable(torch.from_numpy(we2.vectors[id2])).cuda().float()

        # Discriminator A
        X_BA = G_BA(X_B)
        D_A_real = D_A(X_A)
        D_A_fake = D_A(X_BA)

        L_D_A = - torch.mean(log(D_A_real) + log(1 - D_A_fake))

        # Discriminator B
        X_AB = G_AB(X_A)
        D_B_real = D_B(X_B)
        D_B_fake = D_B(X_AB)

        L_D_B = - torch.mean(log(D_B_real) + log(1 - D_B_fake))

        # Total discriminator loss
        D_loss = L_D_A + L_D_B

        D_loss.backward()
        D_solver.step()
        reset_grad()

        # Generator AB
        X_AB = G_AB(X_A)
        D_B_fake = D_B(X_AB)
        X_ABA = G_BA(X_AB)

        L_adv_B = -torch.mean(log(D_B_fake))
        L_recon_A = torch.mean(torch.sum((X_A - X_ABA)**2, 1))
        L_G_AB = L_adv_B + L_recon_A

        # Generator BA
        X_BA = G_BA(X_B)
        D_A_fake = D_A(X_BA)
        X_BAB = G_AB(X_BA)

        L_adv_A = - torch.mean(log(D_A_fake))
        L_recon_B = torch.mean(torch.sum((X_B - X_BAB)**2, 1))
        L_G_BA = L_adv_A + L_recon_B

        # Total generator loss
        G_loss = L_G_AB + L_G_BA

        G_loss.backward()
        G_solver.step()
        reset_grad()


        # Print and plot every now and then
        if it % print_interval == 0:
            W = G_AB.map.weight.data.cpu().numpy()
            print(" GAB_recon_loss:{:.4f} GBA_recon_loss:{:.4f} ||W^T*W - I||:{:.4f}".format(L_recon_A.data[0], L_recon_B.data[0],
                                                                                 np.linalg.norm(
                                                                                     np.dot(W.T, W) - np.identity(d))))

            if (it > 10000) and (G_loss.data[0] < gloss_min):
                gloss_min = G_loss.data[0]
                W = G_AB.map.weight.data.cpu().numpy()
                torch.save(G_AB.state_dict(), 'G_AB_params_min.pkl')
                torch.save(G_BA.state_dict(), 'G_BA_params_min.pkl')
                print("epoch:{} sum_loss:{:.4f}".format(it, G_loss.data[0]))
                print(" GAB_recon_loss:{:.4f} GBA_recon_loss:{:.4f} ||W^T*W - I||:{:.4f}".format(L_recon_A.data[0],L_recon_B.data[0],
                                                                        np.linalg.norm(
                                                                            np.dot(W.T, W) - np.identity(d))))
    torch.save(G_AB.state_dict(), 'G_AB_params_final.pkl')
    torch.save(G_BA.state_dict(), 'G_BA_params_final.pkl')

print('Training time',(time.time() - start_time) / 60, 'min')



G2 = Generator(input_size=g_input_size, output_size=g_output_size).cuda()
G2.load_state_dict(torch.load('G_AB_params_min.pkl'))

d_input_data_all = Variable(torch.from_numpy(we1.vectors)).cuda().float()

transformed_data = G2(d_input_data_all)

we1.transformed_vectors = transformed_data.data.cpu().numpy()
we1.save_transformed_vectors(dataDir + '/UBiLexAT3/data/zh-en/transformed-1' + '.' + 'zh')
print('All running time',(time.time() - start_time) / 60, 'min')
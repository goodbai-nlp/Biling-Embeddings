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

torch.cuda.set_device(1)

g_input_size = 50     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 50    # size of generated output vector
d_input_size = 50   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'


HALF_BATCH_SIZE = 128

# d_learning_rate = 2e-4  # 2e-4
d_learning_rate = 0.001
# g_learning_rate = 2e-4
g_learning_rate = 0.001

optim_betas = (0.9, 0.999)
num_epochs = 500000
print_interval = 100
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

input_noise = 0.2
hidden_noise = 0.5

gloss_min = 100000

TrainNew = True


def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (n, d)))  # Gaussian


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.map = nn.Linear(input_size, output_size,bias=False)

    def forward(self, x):
        x = self.map(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size,output_size)
        self.sigmod = nn.Sigmoid()


    def forward(self, x):
        return self.sigmod(self.map1(x))

# def weight_init(m):
#     # 参数初始化。 可以改成xavier初始化方法
#     class_name=m.__class__.__name__
#     if class_name.find('conv')!=-1:
#         m.weight.data.normal_(0,0.02)
#     if class_name.find('norm')!=-1:
#         m.weight.data.normal_(1.0,0.02)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.01)

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]


G = Generator(input_size=g_input_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_size, output_size=d_output_size)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
criterion2 = nn.CosineSimilarity(dim=1, eps=1e-6)
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

G.cuda()
D.cuda()

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
if TrainNew:
    for epoch in range(num_epochs):

        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()
            id1 = next(we_batches1)
            id2 = next(we_batches2)

            d_input_data = Variable(torch.from_numpy(we1.vectors[id1]))
            d_trg_data = Variable(torch.from_numpy(we2.vectors[id2]))

            #  1A: Train D on real

            d_real_decision = D(d_trg_data.cuda().float())
            d_real_error = criterion(d_real_decision, Variable(torch.ones(HALF_BATCH_SIZE)).cuda())  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_sampler = get_distribution_sampler(1, input_noise)
            gaussian_noise = d_sampler(HALF_BATCH_SIZE)

            d_fake_data = G(d_input_data.cuda().float()).detach()  # detach to avoid training G on these labels,假设G固定

            tmp = d_fake_data.data
            # print(tmp)
            # print(gaussian_noise)
            d_fake_input = Variable(tmp * gaussian_noise.cuda())
            d_fake_decision = D(d_fake_input.cuda().float())
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(HALF_BATCH_SIZE)).cuda())  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()


        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            id1 = next(we_batches1)
            id2 = next(we_batches2)

            d_input_data = Variable(torch.from_numpy(we1.vectors[id1]))
            d_trg_data = Variable(torch.from_numpy(we2.vectors[id2]))

            g_fake_data = G(d_input_data.cuda().float())

            g_recon_data = torch.mm(g_fake_data,G.map.weight.t())

            dg_fake_decision = D(g_fake_data)

            g_error = criterion(dg_fake_decision, Variable(torch.ones(HALF_BATCH_SIZE)).cuda())  # we want to fool, so pretend it's all genuine
            g_recon_loss = 1.0 - torch.mean(criterion2(d_input_data.cuda().float(),g_recon_data))

            loss = g_error+ g_recon_loss
            loss.backward()
            # g_error.backward()
            # g_res_loss.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        # if epoch % print_interval == 0:
        #     print("step: {}  rec_loss:{}  V:{}".format(epoch, g_res_loss.data[0], g_error.data[0]))

        if (epoch> 10000) and (loss.data[0] < gloss_min):
            gloss_min = loss.data[0]
            W = G.map.weight.data.cpu().numpy()
            torch.save(G.state_dict(), 'g_params.pkl')
            print("epoch:{} sum_loss:{}".format(epoch,loss.data[0]))
            print(" recon_gen_loss_val:{}  ||W^T*W - I||:{}".format(g_recon_loss.data[0],
                                                                   np.linalg.norm(np.dot(W.T, W) - np.identity(d))))

G2 = Generator(input_size=g_input_size, output_size=g_output_size).cuda()
G2.load_state_dict(torch.load('g_params.pkl'))

d_input_data_all = Variable(torch.from_numpy(we1.vectors))

transformed_data = G2(d_input_data_all.cuda().float())

we1.transformed_vectors = transformed_data.data.cpu().numpy()
# W = G2.map.weight.data.cpu().numpy()
#
# we1.transformed_vectors = np.dot(we1.vectors, W)
we1.save_transformed_vectors(dataDir + 'transformed-' + '.' + 'zh')
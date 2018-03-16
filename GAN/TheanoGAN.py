#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: TheanoGAN.py
@time: 18-3-14 下午3:40
"""
import sys
import time
import numpy as np
import cPickle
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
import theano
import theano.tensor as T
import lasagne

from embeddings import WordEmbeddings

batch_size = 100
learning_rate = 0.001
n_hidden = 500
n_input = 50
n_noise = 50
recon_weight = 1
num_epochs = 50000
HALF_BATCH_SIZE = 128
optim_betas = (0.9, 0.999)
print_interval = 100
gloss_min = 100000
MODEL_FILENAME = 'model.pkl'
modelID =1

def save_model():
    params_vals = lasagne.layers.get_all_param_values([discriminator.l_out, generator.gen_l_out])
    cPickle.dump(params_vals, open(MODEL_FILENAME, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def load_model():
    params = lasagne.layers.get_all_params([discriminator.l_out, generator.gen_l_out])
    params_vals = cPickle.load(open(MODEL_FILENAME, 'rb'))
    for i, param_val in enumerate(params_vals):
        params[i].set_value(param_val)

def cosine_sim(a_mat, b_mat):
    dp = (a_mat * b_mat).sum(axis=1)
    a_norm = a_mat.norm(2, axis=1)
    b_norm = b_mat.norm(2, axis=1)
    return dp / (a_norm * b_norm)



class Discriminator(object):
    def __init__(self):
        print >> sys.stderr, 'Building computation graph for discriminator...'
        self.input_var = T.matrix('input')
        self.target_var = T.matrix('target')

        self.l_out = self.buildFeedForward(self.input_var)

        self.prediction = lasagne.layers.get_output(self.l_out)
        self.loss = lasagne.objectives.binary_crossentropy(self.prediction, self.target_var).mean()
        # self.accuracy = T.eq(T.ge(self.prediction, 0.5), self.target_var).mean()
        self.params = lasagne.layers.get_all_params(self.l_out, trainable=True)
        self.updates = lasagne.updates.adam(self.loss, self.params, learning_rate=learning_rate)

        self.train_fn = theano.function([self.input_var,self.target_var],[self.loss],updates=self.updates)

    def buildFeedForward(self,input_var=None):
        net = lasagne.layers.InputLayer(shape=(None,n_input),input_var=input_var)
        net = lasagne.layers.dropout(net,0.5)

        nolinear = lasagne.nonlinearities.rectify
        net = lasagne.layers.DenseLayer(net,n_hidden,nonlinearity=nolinear)
        net = lasagne.layers.DenseLayer(net,1,nonlinearity=lasagne.nonlinearities.sigmoid)

        return net

class Generator(object):
    def __init__(self):
        self.gen_input_var = T.matrix('gen_input_var')
        self.gen_l_in = lasagne.layers.InputLayer(shape=(None,n_input),input_var = self.gen_input_var,name = 'gen_l_in')
        self.gen_l_out = lasagne.layers.DenseLayer(self.gen_l_in,num_units=n_input,nonlinearity=None,W=lasagne.init.Orthogonal(),b=None,name='gen_l_out')

        self.dec_l_out = lasagne.layers.DenseLayer(self.gen_l_out, num_units=n_input, nonlinearity=None, W=self.gen_l_out.W.T, b=None, name='dec_l_out')



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

discriminator = Discriminator()
generator = Generator()

generation = lasagne.layers.get_output(generator.gen_l_out)
generation.name = 'generation'

discriminator_prediction = lasagne.layers.get_output(discriminator.l_out, generation, deterministic=True) ##这里的做法
adv_gen_loss = -T.log(discriminator_prediction).mean()
adv_gen_loss.name = 'adv_gen_loss'

reconstruction = lasagne.layers.get_output(generator.dec_l_out)
reconstruction.name = 'reconstruction'

recon_gen_loss = 1.0 - cosine_sim(generator.gen_input_var, reconstruction).mean()
recon_gen_loss.name = 'recon_gen_loss'

gen_loss = adv_gen_loss + recon_weight * recon_gen_loss

gen_params = lasagne.layers.get_all_params(generator.dec_l_out, trainable=True) #这里

gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=0.001)

grad_norm = T.grad(adv_gen_loss, generator.gen_l_out.W).norm(2, axis=1).mean()

print >> sys.stderr, 'Compiling generator...'

gen_train_fn = theano.function([generator.gen_input_var], [gen_loss, recon_gen_loss, adv_gen_loss, generation], updates=gen_updates)

print >> sys.stderr, 'Training...'

X = np.zeros((2 * HALF_BATCH_SIZE, d), dtype=theano.config.floatX)      #注意这里是临时变量
target_mat = np.vstack([np.zeros((HALF_BATCH_SIZE, 1)), np.ones((HALF_BATCH_SIZE, 1))]).astype(theano.config.floatX)

start_time = time.time()
print >> sys.stderr, 'Initial det(W)', np.linalg.det(generator.gen_l_out.W.get_value())

for batch_id in range(1, num_epochs + 1):
    id1 = next(we_batches1)
    id2 = next(we_batches2)
    X[:HALF_BATCH_SIZE] = we1.vectors[id1]
    gen_loss_val, recon_gen_loss_val, adv_gen_loss_val, X_gen = gen_train_fn(X[:HALF_BATCH_SIZE])

    X[:HALF_BATCH_SIZE] = X_gen
    X[HALF_BATCH_SIZE:] = we2.vectors[id2]

    loss_val = discriminator.train_fn(X, target_mat)

    if batch_id > 10000 and gen_loss_val < gloss_min:
        gloss_min = gen_loss_val
        print >> sys.stderr, batch_id, gloss_min
        save_model()
        W = generator.gen_l_out.W.get_value()
        print >> sys.stderr, 'recon_gen_loss_val', recon_gen_loss_val, '||W^T*W - I||', np.linalg.norm(np.dot(W.T, W) - np.identity(d)), 'det(W)', np.linalg.det(W)

        we1.transformed_vectors = np.dot(we1.vectors, W)        #这里换成generator.gen_out试试？
        # we1.transformed_vectors = generation.get_value()
        we1.save_transformed_vectors(dataDir + 'transformed-' + str(modelID) + '.' + 'zh')

print >> sys.stderr, (time.time() - start_time) / 60, 'min'
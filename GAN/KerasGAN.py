#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: KerasGAN.py
@time: 18-3-14 下午8:47
"""
import os
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K

import sys
import numpy as np
from embedding import WordEmbeddings
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
import time
import theano.tensor as T

def cosine_sim(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)

    return y_true * y_pred


class GAN(object):
    def __init__(self):
        self.total_epoch = 100
        self.batch_size = 100
        self.learning_rate = 0.001
        self.n_hidden = 500
        self.n_input = 50
        self.n_noise = 50
        self.recon_weight = 1
        self.num_epochs = 50000
        self.HALF_BATCH_SIZE = 128
        self.optim_betas = (0.9, 0.999)
        self.print_interval = 100
        self.gloss_min = 100000

        optimizer = Adam(self.learning_rate)

        # discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = Input(shape=(self.n_input,))
        fake = self.generator(z)

        self.discriminator.trainable = False        #对于combined模型,只训练generator
        valid = self.discriminator(fake)

        self.combined = Model(z, valid)

        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise_shape = (self.n_input,self.HALF_BATCH_SIZE)
        model = Sequential()
        model.add(Dense(self.n_noise, input_shape=noise_shape,kernel_initializer='Orthogonal' ,activation=None, use_bias=False))
        model.summary()
        z = Input(shape=noise_shape)
        fake = model(z)
        return Model(z, fake)

    def build_discriminator(self):
        emb_shape = (self.n_input,)
        model = Sequential()
        model.add(Dropout(0.5,input_shape=emb_shape))
        model.add(Dense(self.n_hidden,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()
        emb = Input(shape=emb_shape)
        validity = model(emb)

        return Model(emb,validity)


    def train(self,epochs,print_interval=50):

        dataDir = './data/zh-en/'
        rng = check_random_state(0)

        we1 = WordEmbeddings()
        we1.load_from_word2vec(dataDir, 'zh')
        we1.downsample_frequent_words()
        we1.vectors = normalize(we1.vectors)
        we_batches1 = we1.sample_batches(batch_size=self.HALF_BATCH_SIZE, random_state=rng)

        we2 = WordEmbeddings()
        we2.load_from_word2vec(dataDir, 'en')
        we2.downsample_frequent_words()
        we2.vectors = normalize(we2.vectors)
        we_batches2 = we2.sample_batches(batch_size=self.HALF_BATCH_SIZE, random_state=rng)

        assert we1.embedding_dim == we2.embedding_dim
        d = we1.embedding_dim

        for epoch in range(epochs):

            id1 = next(we_batches1)
            id2 = next(we_batches2)

            X_A = we1.vectors[id1]
            X_B = we2.vectors[id2]

            # Generate a half batch of new images
            gen_emb = self.generator.predict(X_A)


            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(X_B, np.ones((self.HALF_BATCH_SIZE, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_emb, np.zeros((self.HALF_BATCH_SIZE, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            valid_y = np.array([1] * self.HALF_BATCH_SIZE)
            g_adv_loss = self.combined.train_on_batch(X_A, valid_y)

            W = self.generator.layers[0].get_weights()[0]
            # print len(self.generator.layers)
            # print W
            recon_emb = T.dot(gen_emb,W.T)

            g_recon_loss = 1.0 - K.mean(cosine_sim(recon_emb, X_A))

            g_loss = g_adv_loss + self.recon_weight*g_recon_loss

            if epoch%print_interval==0:

                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch >10000 and g_loss[0] < self.gloss_min:
                W = self.generator.layers[0].get_weights()
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] ||W^TW-I||:%.4f" % (epoch, d_loss[0], 100 * d_loss[1], g_loss,np.linalg.norm(np.dot(W.T, W) - np.identity(d))))
                self.gloss_min = g_loss[0]
                gen_emb = self.generator.predict(we1.vectors)
                we1.transformed_vectors = gen_emb
                we1.save_transformed_vectors(dataDir + 'transformed-' + str(1) + '.' + 'zh')


if __name__ == '__main__':
    start_time = time.time()
    gan = GAN()
    gan.train(epochs=50000, print_interval=100)
    print('All running time', (time.time() - start_time) / 60, 'min')







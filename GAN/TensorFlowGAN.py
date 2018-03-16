#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: TensorFlowGAN.py
@time: 18-3-13 下午3:18
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from embedding import WordEmbeddings
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
import time
import sys



total_epoch = 100
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

#Descriminator网络输入图片形状
x = tf.placeholder(tf.float32,[None,n_input])
#Generator网络输入的是噪声
z = tf.placeholder(tf.float32,[None,n_noise])

Generator_param={
    # 'gw_1':tf.Variable(tf.random_normal([n_noise,n_input],stddev=0.1))
    'g_w1':tf.get_variable('g_w1',[n_input,n_noise],initializer=tf.orthogonal_initializer())
    # 'g_w2':tf.get_variable('g_w2',[n_noise,n_input],initializer='g_w1'.)
}

Discriminator_param={
    'd_w1':tf.Variable(tf.random_normal([n_input,n_hidden],stddev=0.1)),
    'd_b1':tf.Variable(tf.zeros([n_hidden])),
    'd_w2':tf.Variable(tf.random_normal([n_hidden,1],stddev=0.1)),
    'd_b2':tf.Variable(tf.zeros([1]))
}
def cosine_sim(y_true, y_pred):
    y_true = tf.nn.l2_normalize(y_true, dim=0)
    y_pred = tf.nn.l2_normalize(y_pred, dim=0)

    return y_true * y_pred


def generator(noise_z):
    output = tf.matmul(noise_z,Generator_param['g_w1'])
    recon = tf.matmul(output,Generator_param['g_w1'],transpose_b=True)
    return output,recon

def reconstructor(gen_output):
    output = tf.matmul(gen_output,Generator_param['g_w1'],transpose_b=True)
    return output

def discriminator(inputs):
    # x = tf.nn.dropout(inputs, keep_prob=0.5)
    hidden = tf.nn.relu(tf.nn.dropout(tf.matmul(inputs,Discriminator_param['d_w1'])+Discriminator_param['d_b1'],keep_prob=0.5))
    output = tf.nn.sigmoid(tf.matmul(hidden,Discriminator_param['d_w2'])+Discriminator_param['d_b2'])
    return output

generator_output,recon_output = generator(z)         # fake
# Faker = tf.placeholder(tf.float32,[2*HALF_BATCH_SIZE,n_input])
# Faker[:HALF_BATCH_SIZE] = generator_output

discriminator_pred = discriminator(generator_output) # fake score
discriminator_real = discriminator(x)   # real score
# recon_output = reconstructor(generator_output)

# define losses
g_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_pred, labels = tf.ones_like(discriminator_pred)))
# recon_loss = 1.0 - tf.losses.cosine_distance(z,recon_output,dim=1,reduction='weighted_sum_over_batch_size')
recon_loss = 1.0 - tf.reduce_mean(cosine_sim(z,recon_output))
g_loss = g_adv_loss + recon_weight * recon_loss



d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_real, labels = tf.ones_like(discriminator_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_pred, labels = tf.zeros_like(discriminator_pred)))

tvars = tf.trainable_variables()
# d_vars = [var for var in tvars if 'd_' in var.name]
d_vars = [Discriminator_param['d_w1'],Discriminator_param['d_b1'],Discriminator_param['d_w2'],Discriminator_param['d_b2']]

# g_vars = [var for var in tvars if 'g_' in var.name]
g_vars=[Generator_param['g_w1']]


d_trainer_fake = tf.train.AdamOptimizer(0.001).minimize(d_loss_fake, var_list=d_vars)

d_trainer_real = tf.train.AdamOptimizer(0.001).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.001).minimize(g_loss, var_list=g_vars)


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
start_time = time.time()


TrainNew = int(sys.argv[1])
saver=tf.train.Saver(max_to_keep=3)
with tf.device("/cpu:0"):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        if TrainNew:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(num_epochs):
                id1 = next(we_batches1)
                id2 = next(we_batches2)

                X_A = we1.vectors[id1]
                X_B = we2.vectors[id2]

                # Train discriminator on both real and fake embeddings

                _,__,d_real,d_fake = sess.run([d_trainer_real, d_trainer_fake,d_loss_real,d_loss_fake], feed_dict={x: X_B, z: X_A})
                _, generator_c,g_recon = sess.run([g_trainer, g_adv_loss,recon_loss], feed_dict={z: X_A})

                if epoch % print_interval == 0:
                    W = Generator_param['g_w1'].eval()
                    print('epoch: ', int(epoch / print_interval), '--g_adv_loss: %.4f' % generator_c,'--g_recon_loss:%.4f'%g_recon,
                          '--discriminator_loss: %.4f'% (d_real+d_fake), '||W^T*W-I||: %.4f'%(np.linalg.norm(np.dot(W.T, W) - np.identity(d))))

                if (epoch>10000) and (generator_c < gloss_min):
                    gloss_min = generator_c
                    saver.save(sess,'ckpt/min.ckpt',global_step=epoch+1)

            saver.save(sess, 'ckpt/final.ckpt', global_step=num_epochs)
            print('Training time',(time.time() - start_time) / 60, 'min')

        else:
            model_file=tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess,model_file)
            # saver.restore(sess,'ckpt/min.ckpt-10029')
            res = sess.run(generator_output, feed_dict={z: we1.vectors})
            we1.transformed_vectors = res
            we1.save_transformed_vectors(dataDir + '/UBiLexAT3/data/zh-en/transformed-1' + '.' + 'zh')

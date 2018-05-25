import sys
import torch
import numpy as np
import torch.utils.data
#from properties300 import *
import torch.optim as optim
from util import *
from model import Generator, Discriminator
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random

import copy
from evaluator import Evaluator
from word_translation import get_word_translation_accuracy
torch.cuda.set_device(0)

class TrainerThu:
    def __init__(self, params):
        self.params = params
        self.src_ids, self.tgt_ids = load_npy_two(params.data_dir, 'src_ids.npy', 'tgt_ids.npy', dict=True)
        
    def initialize_exp(self, seed):
        if seed >= 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def weights_init(self, m):  # 正交初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0.01)

    def weights_init3(self, m):  # 单位阵初始化
        if isinstance(m, torch.nn.Linear):
            m.weight.data.copy_(torch.diag(torch.ones(self.params.g_input_size)))

    def train(self, src_emb, tgt_emb):
        params = self.params
        # Load data
        if not os.path.exists(params.data_dir):
            raise "Data path doesn't exists: %s" % params.data_dir

        en = src_emb
        it = tgt_emb
        self.params = _get_eval_params(params)
        params = self.params

        for _ in range(params.num_random_seeds):

            # Create models
            g = Generator(input_size=params.g_input_size, output_size=params.g_output_size)
            d = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size, output_size=params.d_output_size)
            print(d)
            lowest_loss = 1e5
            
            g.apply(self.weights_init3)

            seed = random.randint(0, 1000)
            self.initialize_exp(seed)
            
            loss_fn = torch.nn.BCELoss()
            loss_fn2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #d_optimizer = optim.SGD(d.parameters(), lr=params.d_learning_rate)
            #g_optimizer = optim.SGD(g.parameters(), lr=params.g_learning_rate)
            #d_optimizer = optim.Adam(d.parameters(), lr=params.d_learning_rate)
            d_optimizer = optim.RMSprop(d.parameters(), lr=params.d_learning_rate)
            g_optimizer = optim.Adam(g.parameters(), lr=params.g_learning_rate)

            if torch.cuda.is_available():
                # Move the network and the optimizer to the GPU
                g = g.cuda()
                d = d.cuda()
                loss_fn = loss_fn.cuda()
                loss_fn2 = loss_fn2.cuda()
            # true_dict = get_true_dict(params.data_dir)
            d_acc_epochs = []
            g_loss_epochs = []
            d_loss_epochs = []
            acc_all = []
            d_losses = []
            g_losses = []
            csls_epochs = []
            recon_losses = []
            w_losses = []

            try:
                for epoch in range(params.num_epochs):
                    recon_losses = []
                    w_losses = []
                    start_time = timer()

                    for mini_batch in range(0, params.iters_in_epoch // params.mini_batch_size):
                        hit,total = 0,0
                        for d_index in range(params.d_steps):
                            d_optimizer.zero_grad()  # Reset the gradients
                            d.train()
                            # input, output = self.get_batch_data_fast(en, it, g, detach=True)
                            src_batch, tgt_batch = self.get_batch_data_fast_new(en, it)
                            fake,_ = g(src_batch)
                            fake = fake.detach()
                            real = tgt_batch
                            # input = torch.cat([fake, real], 0)
                            input = torch.cat([real, fake], 0)
                            output = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())

                            output[:params.mini_batch_size] = 1 - params.smoothing
                            output[params.mini_batch_size:] = params.smoothing

                            pred = d(input)
                            d_loss = loss_fn(pred, output)
                            d_loss.backward()  # compute/store gradients, but don't change params
                            d_losses.append(d_loss.data.cpu().numpy())
                            discriminator_decision = pred.data.cpu().numpy()
                            hit += np.sum(discriminator_decision[:params.mini_batch_size] >= 0.5)
                            hit += np.sum(discriminator_decision[params.mini_batch_size:] < 0.5)
                            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                            # Clip weights
                            _clip(d, params.clip_value)

                            sys.stdout.write("[%d/%d] :: Discriminator Loss: %f \r" % (
                                mini_batch, params.iters_in_epoch // params.mini_batch_size,
                                np.asscalar(np.mean(d_losses[-1000:]))))
                            sys.stdout.flush()

                        total += 2 * params.mini_batch_size * params.d_steps

                        for g_index in range(params.g_steps):
                            # 2. Train G on D's response (but DO NOT train D on these labels)
                            g_optimizer.zero_grad()
                            d.eval()
                            src_batch, tgt_batch = self.get_batch_data_fast_new(en, it)
                            fake, recon = g(src_batch)
                            real = tgt_batch
                            output = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
                            output[:params.mini_batch_size] = 1 - params.smoothing
                            output[params.mini_batch_size:] = params.smoothing

                            pred = d(fake)
                            output2 = to_variable(torch.FloatTensor(params.mini_batch_size).zero_())
                            output2 = output2+1-params.smoothing
                            
                            recon_loss = 1.0 - torch.mean(loss_fn2(src_batch,recon))
                            g_loss = loss_fn(pred, output2) + params.recon_weight * recon_loss
                            g_loss.backward()
                            
                            g_losses.append(g_loss.data.cpu().numpy())
                            recon_losses.append(recon_loss.data.cpu().numpy())

                            g_optimizer.step()  # Only optimizes G's parameters
                            #self.orthogonalize(g.map1.weight.data)

                            sys.stdout.write("[%d/%d] ::                                     Generator Loss: %f \r" % (
                                mini_batch, params.iters_in_epoch // params.mini_batch_size,
                                np.asscalar(np.mean(g_losses[-1000:]))))
                            sys.stdout.flush()

                        acc_all.append(hit / total)
                        
                        if epoch > params.threshold:
                            if lowest_loss > float(g_loss.data):
                                lowest_loss = float(g_loss.data)
                                W = g.map1.weight.data.cpu().numpy()
                                w_losses.append(np.linalg.norm(np.dot(W.T, W) - np.identity(params.g_input_size)))

                                X_Z = g(src_emb.weight)[0].data
                                Y_Z = tgt_emb.weight.data

                                mstart_time = timer()
                                for method in [params.dico_method]:
                                    results = get_word_translation_accuracy(
                                        'en', self.src_ids, X_Z,
                                        'zh', self.tgt_ids, Y_Z,
                                        method=method,
                                        path = params.data_dir+params.validation_file
                                    )
                                    acc = results[0][1]
                                    #print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
                                    #print('Method:{} score:{:.4f}'.format(method,acc))

                                    torch.save(g.state_dict(),
                                               'tune/best/G_seed{}_epoch_{}_batch_{}_mf_{}_p@1_{:.3f}.t7'.format(seed,epoch,mini_batch,params.most_frequent_sampling_size,acc))
 
                        '''
                        if mini_batch % 500==0:
                            #d_acc_epochs.append(hit / total)
                            #d_loss_epochs.append(np.asscalar(np.mean(d_losses)))
                            #g_loss_epochs.append(np.asscalar(np.mean(g_losses)))
                            if epoch > params.threshold:
                                W = g.map1.weight.data.cpu().numpy()
                                w_loss = np.linalg.norm(np.dot(W.T, W) - np.identity(params.g_input_size))
                                #print("D_acc:{:.3f} d_loss:{:.3f} g_loss:{:.3f} w_loss:{:.2f} ".format(hit / total,np.asscalar(np.mean(d_losses)),np.asscalar(np.mean(g_losses)),w_loss))
                                #print("D_acc:{:.3f} d_loss:{:.3f} g_loss:{:.3f} w_loss:{:.2f}".format(hit / total,d_loss.data[0],g_loss.data[0],w_loss))

                                X_Z = g(src_emb.weight)[0].data
                                Y_Z = tgt_emb.weight.data

                                mstart_time = timer()
                                for method in [params.dico_method]:
                                    results = get_word_translation_accuracy(
                                        'en', self.src_ids, X_Z,
                                        'zh', self.tgt_ids, Y_Z,
                                        method=method,
                                        path = params.validation_file
                                    )
                                    acc = results[0][1]
                                    #print('epoch:{} Method:{} score:{:.4f}'.format(mini_batch,method,acc))
                                    torch.save(g.state_dict(),
                                       'tune/thu/G_seed{}_epoch_{}_mf_{}_p@1_{:.3f}.t7'.format(seed,mini_batch,params.most_frequent_sampling_size,acc))
                        '''
                    X_Z = g(src_emb.weight)[0].data
                    Y_Z = tgt_emb.weight.data
                    print("Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f},Generator Loss: {:.5f}, Time elapsed {:.2f}mins".format(epoch,np.asscalar(np.mean(d_losses[-1562:])), hit / total,np.asscalar(np.mean(g_losses[-1562:])),(timer() - start_time) / 60))

                    mstart_time = timer()
                    for method in [params.dico_method]:
                        results = get_word_translation_accuracy(
                            'en', self.src_ids, X_Z,
                            'zh', self.tgt_ids, Y_Z,
                            method=method,
                            path = params.data_dir+params.validation_file
                        )
                        acc = results[0][1]
                        print('epoch:{} Method:{} score:{:.4f}'.format(epoch,method,acc))
                        torch.save(g.state_dict(),
                            'tune/G_seed{}_epoch_{}_mf_{}_p@1_{:.3f}.t7'.format(seed,epoch,params.most_frequent_sampling_size,acc))

                
                # Save the plot for discriminator accuracy and generator loss
                fig = plt.figure()
                plt.plot(range(0, len(acc_all)), acc_all, color='b', label='discriminator')
                plt.ylabel('D_accuracy_all')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune/D_acc_all.png')

                fig = plt.figure()
                plt.plot(range(0, len(d_losses)), d_losses, color='b', label='discriminator')
                plt.ylabel('D_loss_all')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune/D_loss_all.png')

                fig = plt.figure()
                plt.plot(range(0, len(g_losses)), g_losses, color='b', label='discriminator')
                plt.ylabel('G_loss_all')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune/G_loss_all.png')

                fig = plt.figure()
                plt.plot(range(0, len(w_losses)), w_losses, color='b', label='discriminator')
                plt.ylabel('||W^T*W - I||')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune/W^TW.png')

                plt.close('all')

            except KeyboardInterrupt:
                print("Interrupted.. saving model !!!")
                torch.save(g.state_dict(), 'tune/g_model_interrupt.t7')
                torch.save(d.state_dict(), 'tune/d_model_interrupt.t7')
                exit()

        return g
    def orthogonalize(self, W):
        params = self.params
        W.copy_((1 + params.beta) * W - params.beta * W.mm(W.transpose(0, 1).mm(W)))

    def get_batch_data_fast_new(self,en,it):
        params = self.params
        random_en_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        random_it_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        #random_en_indices = torch.LongTensor(params.mini_batch_size).random_(min(len(en.weight.data),params.most_frequent_sampling_size))
        #random_it_indices = torch.LongTensor(params.mini_batch_size).random_(min(len(it.weight.data),params.most_frequent_sampling_size))
        en_batch = en(to_variable(random_en_indices))
        it_batch = it(to_variable(random_it_indices))
        return en_batch,it_batch

def _init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)


def _clip(d, clip):
    if clip > 0:
        for x in d.parameters():
            x.data.clamp_(-clip, clip)

def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1]
    params.methods = ['csls']
    params.models = ['adv']
    params.refine = ['without-ref']

    # params.dico_method = "csls_knn_10"
    params.dico_method = "nn"
    # params.dico_build = "S2T&T2S"
    params.dico_build = "S2T"
    params.dico_max_rank = 10000
    params.dico_max_size = 10000
    params.dico_min_size = 0
    params.dico_threshold = 0
    
    params.threshold = 5
    params.cuda = True

    params.recon_weight = 1
    params.d_learning_rate = 0.0003
    params.g_learning_rate = 0.0005
    params.d_step = 1
    return params

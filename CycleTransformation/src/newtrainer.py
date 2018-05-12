import sys
import torch
import numpy as np
import torch.utils.data
# from properties import *
import torch.optim as optim
from util import *
from newmodel import Generator, Discriminator
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
from datetime import timedelta
import json
import copy
from evaluator import Evaluator
torch.cuda.set_device(1)

class TrainerDual:
    def __init__(self, params):
        self.params = params
        
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

    def weights_init2(self, m):  # xavier_normal 初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal(m.weight)
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
        
        params = _get_eval_params(params)
        eval = Evaluator(params, src_emb.weight.data, tgt_emb.weight.data, use_cuda=True)

        for _ in range(params.num_random_seeds):

            # Create models
            G_AB = Generator(params, [300, 300],True)
            G_BA = Generator(params, [300, 300],True)
            D_A = Discriminator(params, [300, 2500,2500, 1])
            D_B = Discriminator(params, [300, 2500,2500, 1])
            nets = [G_AB, G_BA, D_A, D_B]


            seed = random.randint(0, 1000)
            # init_xavier(g)
            # init_xavier(d)
            self.initialize_exp(seed)

            G_params = list(G_AB.parameters()) + list(G_BA.parameters())
            D_params = list(D_A.parameters()) + list(D_B.parameters())

            # self.G_solver = optim.Adam(G_params, lr=params.d_learning_rate, betas=params.optim_betas)
            # self.D_solver = optim.Adam(D_params, lr=params.g_learning_rate, betas=params.optim_betas)

            # Define loss function and optimizers
            loss_fn = torch.nn.BCELoss()
            loss_fn2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            G_optimizer = optim.SGD(G_params, lr=params.d_learning_rate)
            D_optimizer = optim.SGD(D_params, lr=params.g_learning_rate)

            if torch.cuda.is_available():
                # Move the network and the optimizer to the GPU
                for net in nets:
                    net.cuda()
                loss_fn = loss_fn.cuda()
                loss_fn2 = loss_fn2.cuda()

            G_AB.apply(self.weights_init)  # 可更改G初始化方式
            G_BA.apply(self.weights_init)  # 可更改G初始化方式
            D_A.apply(self.weights_init2)
            D_B.apply(self.weights_init2)

            # true_dict = get_true_dict(params.data_dir)
            D_A_acc_epochs = []
            D_B_acc_epochs = []
            D_A_loss_epochs = []
            D_B_loss_epochs = []
            G_AB_loss_epochs = []
            G_BA_loss_epochs = []
            G_AB_recon_epochs = []
            G_BA_recon_epochs = []

            acc_epochs = []
            csls_epochs = []

            # logs for plotting later
            log_file = open("log_src_tgt.txt", "w")     # Being overwritten in every loop, not really required
            log_file.write("epoch, dis_loss, dis_acc, g_loss\n")

            try:
                for epoch in range(params.num_epochs):
                    D_A_losses = []
                    D_B_losses = []
                    G_AB_losses = []
                    G_AB_recon = []
                    G_BA_losses = []
                    G_BA_recon = []
                    d_losses = []
                    g_losses = []
                    hit_A = 0
                    hit_B = 0
                    total = 0
                    start_time = timer()

                    label_D = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
                    label_D[:params.mini_batch_size] = 1 - params.smoothing
                    label_D[params.mini_batch_size:] = params.smoothing

                    label_G = to_variable(torch.FloatTensor(params.mini_batch_size).zero_())
                    label_G = label_G + 1 - params.smoothing

                    for mini_batch in range(0, params.iters_in_epoch // params.mini_batch_size):
                        for d_index in range(params.d_steps):
                            D_optimizer.zero_grad()  # Reset the gradients
                            D_A.train()
                            D_B.train()

                            X_A, X_B = self.get_batch_data_fast_new(en, it)

                            # Discriminator A
                            X_BA = G_BA(X_B).detach()

                            input = torch.cat([X_A, X_BA], 0)

                            pred_A = D_A(input)
                            D_A_loss = loss_fn(pred_A, label_D)

                            # Discriminator B
                            X_AB = G_AB(X_A).detach()
                            input = torch.cat([X_B,X_AB],0)
                            pred_B = D_B(input)
                            D_B_loss = loss_fn(pred_B, label_D)

                            D_loss = D_A_loss + D_B_loss

                            D_loss.backward()               # compute/store gradients, but don't change params
                            d_losses.append(D_loss.data.cpu().numpy())
                            D_A_losses.append(D_A_loss.data.cpu().numpy())
                            D_B_losses.append(D_B_loss.data.cpu().numpy())

                            discriminator_decision_A = pred_A.data.cpu().numpy()
                            hit_A += np.sum(discriminator_decision_A[:params.mini_batch_size] >= 0.5)
                            hit_A += np.sum(discriminator_decision_A[params.mini_batch_size:] < 0.5)
                            discriminator_decision_B = pred_B.data.cpu().numpy()
                            hit_B += np.sum(discriminator_decision_B[:params.mini_batch_size] >= 0.5)
                            hit_B += np.sum(discriminator_decision_B[params.mini_batch_size:] < 0.5)

                            D_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                            # Clip weights
                            _clip(D_A, params.clip_value)
                            _clip(D_B, params.clip_value)

                            sys.stdout.write("[%d/%d] :: Discriminator Loss: %f \r" % (
                                mini_batch, params.iters_in_epoch // params.mini_batch_size, np.asscalar(np.mean(d_losses))))
                            sys.stdout.flush()

                        total += 2 * params.mini_batch_size * params.d_steps

                        for g_index in range(params.g_steps):
                            # 2. Train G on D's response (but DO NOT train D on these labels)
                            G_optimizer.zero_grad()
                            D_A.eval()
                            D_B.eval()
                            # input, output = self.get_batch_data_fast(en, it, g, detach=False)
                            X_A, X_B = self.get_batch_data_fast_new(en, it)

                            # Generator AB
                            X_AB = G_AB(X_A)
                            D_B_fake = D_B(X_AB)

                            L_adv_B = loss_fn(D_B_fake,label_G)

                            X_ABA = G_BA(X_AB)
                            L_recon_A = 1.0 - torch.mean(loss_fn2(X_A, X_ABA))
                            L_G_AB = L_adv_B + params.recon_weight*L_recon_A

                            # Generator BA
                            X_BA = G_BA(X_B)
                            D_A_fake = D_A(X_BA)
                            X_BAB = G_AB(X_BA)
                            L_adv_A = loss_fn(D_A_fake,label_G)
                            L_recon_B = 1.0 - torch.mean(loss_fn2(X_B, X_BAB))

                            L_G_BA = L_adv_A + params.recon_weight*L_recon_B
                            G_loss = L_G_AB + L_G_BA
                            G_loss.backward()

                            g_losses.append(G_loss.data.cpu().numpy())
                            G_AB_losses.append(L_G_AB.data.cpu().numpy())
                            G_BA_losses.append(L_G_BA.data.cpu().numpy())
                            G_AB_recon.append(L_recon_A.data.cpu().numpy())
                            G_BA_recon.append(L_recon_B.data.cpu().numpy())

                            G_optimizer.step()  # Only optimizes G's parameters

                            # Orthogonalize
                            # self.orthogonalize(g.map1.weight.data)

                            sys.stdout.write("[%d/%d] ::                                     Generator Loss: %f \r" % (
                                mini_batch, params.iters_in_epoch // params.mini_batch_size, np.asscalar(np.mean(g_losses))))
                            sys.stdout.flush()


                    '''for each epoch'''
                    D_A_acc_epochs.append(hit_A / total)
                    D_B_acc_epochs.append(hit_B / total)
                    G_AB_loss_epochs.append(np.asscalar(np.mean(G_AB_losses)))
                    G_BA_loss_epochs.append(np.asscalar(np.mean(G_BA_losses)))
                    D_A_loss_epochs.append(np.asscalar(np.mean(D_A_losses)))
                    D_B_loss_epochs.append(np.asscalar(np.mean(D_B_losses)))
                    G_AB_recon_epochs.append(np.asscalar(np.mean(G_AB_recon)))
                    G_BA_recon_epochs.append(np.asscalar(np.mean(G_BA_recon)))

                    print("Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f}, Generator Loss: {:.5f}, Time elapsed {:.2f} mins".
                          format(epoch, np.asscalar(np.mean(d_losses)), 0.5*(hit_A+hit_B) / total, np.asscalar(np.mean(g_losses)),
                                 (timer() - start_time) / 60))

                    # lr decay
                    g_optim_state = G_optimizer.state_dict()
                    old_lr = g_optim_state['param_groups'][0]['lr']
                    g_optim_state['param_groups'][0]['lr'] = max(old_lr * params.lr_decay, params.lr_min)
                    G_optimizer.load_state_dict(g_optim_state)
                    print("Changing the learning rate: {} -> {}".format(old_lr, g_optim_state['param_groups'][0]['lr']))
                    d_optim_state = D_optimizer.state_dict()
                    d_optim_state['param_groups'][0]['lr'] = max(d_optim_state['param_groups'][0]['lr'] * params.lr_decay, params.lr_min)
                    D_optimizer.load_state_dict(d_optim_state)

                    if (epoch + 1) % params.print_every == 0:
                        # No need for discriminator weights
                        # torch.save(d.state_dict(), 'discriminator_weights_en_es_{}.t7'.format(epoch))
                        all_precisions = eval.get_all_precisions(G_AB(src_emb.weight).data)
                        # csls = eval.calc_unsupervised_criterion(g(src_emb.weight)[0].data)

                        #print(json.dumps(all_precisions))
                        # p_1 = all_precisions['validation']['adv']['without-ref']['nn'][1]
                        p_1 = all_precisions['validation']['adv']['without-ref']['csls'][1]

                        log_file.write("{},{:.5f},{:.5f},{:.5f}\n".format(epoch + 1, np.asscalar(np.mean(d_losses)), hit_A / total, np.asscalar(np.mean(g_losses))))
                        log_file.write(str(all_precisions) + "\n")
                        # Saving generator weights
                        torch.save(G_AB.state_dict(), 'tune2/G_AB_seed_{}_mf_{}_lr_{}_p@1_{:.3f}.t7'.format(seed, params.most_frequent_sampling_size, params.g_learning_rate, p_1))

                        acc_epochs.append(p_1)
                        # csls_epochs.append(csls)

                # Save the plot for discriminator accuracy and generator loss
                fig = plt.figure()
                plt.plot(range(0, params.num_epochs), D_A_acc_epochs, color='b', label='D_A')
                plt.plot(range(0, params.num_epochs), D_B_acc_epochs, color='r', label='D_B')
                plt.ylabel('D_accuracy')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune2/seed_{}_D_acc.png'.format(seed))

                fig = plt.figure()
                plt.plot(range(0, params.num_epochs), D_A_loss_epochs, color='b', label='D_A')
                plt.plot(range(0, params.num_epochs), D_B_loss_epochs, color='r', label='D_B')
                plt.ylabel('D_losses')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune2/seed_{}_D_loss.png'.format(seed))

                fig = plt.figure()
                plt.plot(range(0, params.num_epochs), G_AB_loss_epochs, color='b', label='G_AB')
                plt.plot(range(0, params.num_epochs), G_BA_loss_epochs, color='r', label='G_BA')
                plt.ylabel('G_losses')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune2/seed_{}_G_loss.png'.format(seed))

                fig = plt.figure()
                plt.plot(range(0, params.num_epochs), G_AB_recon_epochs, color='b', label='G_AB')
                plt.plot(range(0, params.num_epochs), G_BA_recon_epochs, color='r', label='G_BA')
                plt.ylabel('G_recon_loss')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune2/seed_{}_G_recon.png'.format(seed))

                fig = plt.figure()
                plt.plot(range(0, params.num_epochs), acc_epochs, color='b', label='trans_acc')
                plt.ylabel('trans_acc')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('tune2/seed_{}_trans_acc.png'.format(seed))

                plt.close('all')
            except KeyboardInterrupt:
                print("Interrupted.. saving model !!!")
                torch.save(G_AB.state_dict(), 'g_model_interrupt.t7')
                torch.save(D_B.state_dict(), 'd_model_interrupt.t7')
                log_file.close()
                exit()

            log_file.close()

        return G_AB

    def orthogonalize(self, W):
        params = self.params
        W.copy_((1 + params.beta) * W - params.beta * W.mm(W.transpose(0, 1).mm(W)))

    def get_batch_data_fast(self, en, it, g, detach=False):
        params = self.params
        random_en_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        random_it_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        en_batch = en(to_variable(random_en_indices))
        it_batch = it(to_variable(random_it_indices))
        fake = g(en_batch)
        if detach:
            fake = fake.detach()
        real = it_batch
        input = torch.cat([fake, real], 0)
        output = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
        output[:params.mini_batch_size] = 1 - params.smoothing
        output[params.mini_batch_size:] = params.smoothing
        return input, output

    def get_batch_data(self, en, it, g, detach=False):
        params = self.params
        random_en_indices = np.random.permutation(params.most_frequent_sampling_size)
        random_it_indices = np.random.permutation(params.most_frequent_sampling_size)
        en_batch = en[random_en_indices[:params.mini_batch_size]]
        it_batch = it[random_it_indices[:params.mini_batch_size]]
        fake = g(to_variable(to_tensor(en_batch)))
        if detach:
            fake = fake.detach()
        real = to_variable(to_tensor(it_batch))
        input = torch.cat([fake, real], 0)
        output = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
        output[:params.mini_batch_size] = 1 - params.smoothing   # As per fb implementation
        output[params.mini_batch_size:] = params.smoothing
        return input, output

    def get_batch_data_fast_new(self,en,it):

        params = self.params
        random_en_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        random_it_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
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
    return params

import sys
import torch
import numpy as np
import torch.utils.data
# from properties import *
import torch.optim as optim
from util import *
from model import Generator, Discriminator
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
from word_translation import get_word_translation_accuracy

torch.cuda.set_device(3)


class TrainerFb:
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

        params = _get_eval_params(params)
        #eval = Evaluator(params, src_emb.weight.data, tgt_emb.weight.data, use_cuda=True)


        for _ in range(params.num_random_seeds):
            # Create models
            g = Generator(input_size=params.g_input_size, output_size=params.g_output_size)
            d = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                              output_size=params.d_output_size)

            g.apply(self.weights_init3)
            seed = random.randint(0, 1000)
            # init_xavier(g)
            # init_xavier(d)
            self.initialize_exp(seed)

            # Define loss function and optimizers
            loss_fn = torch.nn.BCELoss()
            loss_fn2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            d_optimizer = optim.SGD(d.parameters(), lr=params.d_learning_rate)
            g_optimizer = optim.SGD(g.parameters(), lr=params.g_learning_rate)

            if torch.cuda.is_available():
                # Move the network and the optimizer to the GPU
                g = g.cuda()
                d = d.cuda()
                loss_fn = loss_fn.cuda()
                loss_fn2 = loss_fn2.cuda()
            # true_dict = get_true_dict(params.data_dir)
            d_acc_epochs = []
            g_loss_epochs = []
            #d_losses = []
            #g_losses = []
            #w_losses = []
            acc_epochs = []
            csls_epochs = []
            best_valid_metric = -1
            lowest_loss = 1e5
            # logs for plotting later
            log_file = open("log_src_tgt.txt", "w")  # Being overwritten in every loop, not really required
            log_file.write("epoch, dis_loss, dis_acc, g_loss\n")

            try:
                for epoch in range(params.num_epochs):
                    hit = 0
                    total = 0
                    start_time = timer()
                    d_losses = []
                    g_losses = []
                    w_losses = []

                    for mini_batch in range(0, params.iters_in_epoch // params.mini_batch_size):
                        for d_index in range(params.d_steps):
                            d_optimizer.zero_grad()  # Reset the gradients
                            d.train()
                            src_batch, tgt_batch = self.get_batch_data_fast_new(en, it)
                            fake, _ = g(src_batch)
                            fake = fake.detach()
                            real = tgt_batch

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
                                np.asscalar(np.mean(d_losses))))
                            sys.stdout.flush()

                        total += 2 * params.mini_batch_size * params.d_steps

                        for g_index in range(params.g_steps):
                            # 2. Train G on D's response (but DO NOT train D on these labels)
                            g_optimizer.zero_grad()
                            d.eval()
                            # input, output = self.get_batch_data_fast(en, it, g, detach=False)
                            src_batch, tgt_batch = self.get_batch_data_fast_new(en, it)
                            fake, recon = g(src_batch)
                            real = tgt_batch
                            # input = torch.cat([fake, real], 0)
                            # input = torch.cat([real, fake], 0)
                            output = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
                            output[:params.mini_batch_size] = 1 - params.smoothing
                            output[params.mini_batch_size:] = params.smoothing

                            # pred = d(input)
                            pred = d(fake)
                            output2 = to_variable(torch.FloatTensor(params.mini_batch_size).zero_())
                            output2 = output2 + 1 - params.smoothing
                            # g_loss = loss_fn(pred, 1 - output)
                            # g_loss = loss_fn(pred, 1- output) +  1.0 - torch.mean(loss_fn2(src_batch,recon))
                            g_loss = loss_fn(pred, output2)
                            #g_loss = loss_fn(pred, output2) + params.rwcon_weight*(1.0 -
                                            #torch.mean(loss_fn2(src_batch,recon)))

                            g_loss.backward()
                            g_losses.append(g_loss.data.cpu().numpy())
                            g_optimizer.step()  # Only optimizes G's parameters

                            # Orthogonalize
                            self.orthogonalize(g.map1.weight.data)

                            sys.stdout.write("[%d/%d] ::                                     Generator Loss: %f \r" % (
                                mini_batch, params.iters_in_epoch // params.mini_batch_size,
                                np.asscalar(np.mean(g_losses))))
                            sys.stdout.flush()

                        if epoch > params.threshold:
                            if lowest_loss > float(g_loss.data):
                                lowest_loss = float(g_loss.data)
                                W = g.map1.weight.data.cpu().numpy()
                                w_losses.append(np.linalg.norm(np.dot(W.T, W) - np.identity    (params.g_input_size)))
                                for method in ['nn']:
                                    results = get_word_translation_accuracy(
                                        'en', self.src_ids, g(src_emb.weight)[0].data,
                                        'it', self.tgt_ids, tgt_emb.weight.data,
                                        method=method,
                                        path = params.validation_file
                                    )
                                    acc = results[0][1]
                                torch.save(g.state_dict(),
                                   'tune0/thu/g_seed_{}_epoch_{}_batch_{}_p@1_{:.3f}.t7'.format(seed,epoch,mini_batch,acc))


                    '''for each epoch'''
                    d_acc_epochs.append(hit / total)
                    g_loss_epochs.append(np.asscalar(np.mean(g_losses)))
                    print(
                        "Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f}, Generator Loss: {:.5f}, Time elapsed {:.2f} mins".
                        format(epoch, np.asscalar(np.mean(d_losses)), hit / total, np.asscalar(np.mean(g_losses)),
                               (timer() - start_time) / 60))



                    if (epoch + 1) % params.print_every == 0:
                        # No need for discriminator weights
                        # torch.save(d.state_dict(), 'discriminator_weights_en_es_{}.t7'.format(epoch))
                        mstart_time = timer()
                        #for method in ['csls_knn_10']:
                        for method in ['nn']:
                            results = get_word_translation_accuracy(
                                'en', self.src_ids, g(src_emb.weight)[0].data,
                                'it', self.tgt_ids, tgt_emb.weight.data,
                                method=method,
                                path = params.validation_file
                            )
                            acc = results[0][1]
                            print('{} takes {:.2f}s'.format(method, timer() - mstart_time))


                        # all_precisions = eval.get_all_precisions(g(src_emb.weight)[0].data)
                        csls = 0
                        # csls = eval.calc_unsupervised_criterion(g(src_emb.weight)[0].data)
                        #csls = eval.dist_mean_cosine(g(src_emb.weight)[0].data, tgt_emb.weight.data)
                        # print(json.dumps(all_precisions))
                        # p_1 = all_precisions['validation']['adv']['without-ref']['nn'][1]

                        log_file.write(
                            "{},{:.5f},{:.5f},{:.5f}\n".format(epoch + 1, np.asscalar(np.mean(d_losses)), hit / total,
                                                               np.asscalar(np.mean(g_losses))))
                        # log_file.write(str(all_precisions) + "\n")
                        print('Method:csls_knn_10 score:{:.4f}'.format(acc))
                        # Saving generator weights
                        torch.save(g.state_dict(),
                                   'tune0/generator_weights_src_tgt_seed_{}_mf_{}_lr_{}_p@1_{:.3f}.t7'.format(seed,params.most_frequent_sampling_size,params.g_learning_rate,acc))
                        if csls > best_valid_metric:
                            best_valid_metric = csls
                            torch.save(g.state_dict(),
                                   'tune0/best/generator_weights_src_tgt_seed_{}_mf_{}_lr_{}_p@1_{:.3f}.t7'.format(seed,params.most_frequent_sampling_size,params.g_learning_rate,acc))

                        acc_epochs.append(acc)
                        csls_epochs.append(csls)



                # Save the plot for discriminator accuracy and generator loss
                fig = plt.figure()
                plt.plot(range(0, params.num_epochs), d_acc_epochs, color='b', label='discriminator')
                plt.plot(range(0, params.num_epochs), g_loss_epochs, color='r', label='generator')
                plt.ylabel('accuracy/loss')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('d_g.png')

            except KeyboardInterrupt:
                print("Interrupted.. saving model !!!")
                torch.save(g.state_dict(), 'g_model_interrupt.t7')
                torch.save(d.state_dict(), 'd_model_interrupt.t7')
                log_file.close()
                exit()

            log_file.close()

        return g

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
        output[:params.mini_batch_size] = 1 - params.smoothing  # As per fb implementation
        output[params.mini_batch_size:] = params.smoothing
        return input, output

    def get_batch_data_fast_new(self, en, it):
        params = self.params
        random_en_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        random_it_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        en_batch = en(to_variable(random_en_indices))
        it_batch = it(to_variable(random_it_indices))
        return en_batch, it_batch


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

    params.dico_method="csls_knn_10"
    params.dico_build="S2T&T2S"
    params.dico_max_rank = 15000
    params.dico_max_size = 0
    params.dico_min_size = 0
    params.dico_threshold = 0
    params.cuda = True
    params.threshold = 10
    return params

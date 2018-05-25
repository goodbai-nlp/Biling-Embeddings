#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: myembedding.py
@time: 18-3-31 下午8:43
"""

import numpy as np
import io
import sys
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from logging import getLogger
import torch

logger = getLogger()

class WordEmbeddings(object):
    def __init__(self):
        self.num_words = 0
        self.total_count = 0
        self.words = []
        self.embedding_dim = 0
        self.vectors = []
        self.transformed_vectors = np.zeros((0, 0))
        self.counts = np.zeros(0, dtype=int)
        self.probs = np.zeros(0)
        self.word_dict = dict([])
        self.word2id = {}
        self.id2word = {}
        self.max_vocab = 200000
        self.most_frequent = 75000

    def load_from_word2vec(self, dataDir, lang):
        vocab_file = dataDir + 'vocab-freq.' + lang
        vec_file = dataDir + 'word2vec.' + lang

        vec_fs = open(vec_file)
        line = vec_fs.readline()
        tokens = line.split()
        self.num_words = int(tokens[0])
        self.embedding_dim = int(tokens[1])
        self.vectors = np.zeros((self.num_words, self.embedding_dim))
        self.counts = np.zeros(self.num_words, dtype=int)
        self.probs = np.ones(self.num_words)
        for i, line in enumerate(vec_fs):
            tokens = line.split()
            word = tokens[0]
            self.words.append(word)
            self.word_dict[word] = i
            self.vectors[i] = [float(x) for x in tokens[1:]]
#                self.vectors = normalize(self.vectors)
        vec_fs.close()

        vocab_fs = open(vocab_file)
        for line in vocab_fs:
            tokens = line.split()
            word, count = tokens[0], int(tokens[1])
            i = self.word_dict[word]
            self.counts[i] = count
        vocab_fs.close()
        self.total_count = self.counts.sum()
        self.probs *= self.counts
        self.probs /= self.total_count

    def load_from_word2vec_new(self,dataFile):
        vectors = []
        with io.open(dataFile, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:

            for i, line in enumerate(f):
                if i == 0:
                    split = line.split()
                    assert len(split) == 2
                    self.embedding_dim = int(split[1])
                    # print('embedding dim',self.embedding_dim)
                else:
                    word, vect = line.rstrip().split(' ', 1)
                    word = word.lower()
                    vect = np.fromstring(vect, sep=' ')
                    if np.linalg.norm(vect) == 0:       # avoid all 0 embeddings
                        vect[0] = 0.01
                    if word in self.word2id:
                        logger.info('Warning: Word "%s" found twice in embedding file'%(word))
                        # print('Warning: Word "%s" found twice in embedding file'%(word))
                    else:
                        if not vect.shape[0] == self.embedding_dim:
                            logger.info('Warning: Invalid dimension (%i) for word "%s" in line %i.'%(vect.shape[0],word,i) )
                            print('Warning: Invalid dimension (%i) for word "%s" in line %i.'%(vect.shape[0],word,i) )

                            continue

                        self.word2id[word] = len(self.word2id)
                        self.words.append(word)
                        vectors.append(vect[None])
                if len(self.word2id)>= self.max_vocab:
                    break

        assert len(self.word2id) == len(vectors)
        print("Loaded %i pre-trained word embeddings" % len(vectors))
        print("All {} words".format(len(self.words)))

        # compute new vocabulary / embeddings
        self.num_words = len(self.word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vectors = np.concatenate(vectors, 0)

    def save_transformed_vectors(self, filename):
        print("number of embeddings {}".format(len(self.transformed_vectors)))
        with io.open(filename, 'w',encoding='utf-8') as fout:
            fout.write(str(self.num_words) + ' ' + str(self.embedding_dim) + '\n')
            for i in range(self.num_words):
                fout.write(self.words[i] + ' ' + ' '.join(str(x) for x in self.transformed_vectors[i]) + '\n')


    def downsample_frequent_words(self, frequency_threshold=1e-3):
        threshold_count = float(frequency_threshold * self.total_count)
        self.probs = (np.sqrt(self.counts / threshold_count) + 1) * (threshold_count / self.counts)
        self.probs = np.maximum(self.probs, 1.0)    #Zm: Originally maximum, which upsamples rare words
        self.probs *= self.counts
        self.probs /= self.probs.sum()

    def sample_batches(self, batch_size=1, train_set_ids=None, random_state=0):
        rng = check_random_state(random_state)
        if train_set_ids != None:
            a = train_set_ids
            p = self.probs[train_set_ids]
            p /= p.sum()
        else:
            a = self.num_words
            p = self.probs

        while True:
            rv = rng.choice(a, size=batch_size, replace=True, p=p)
            yield rv

    def uniform_sample_batches(self, batch_size=1, random_state=0):
        rng = check_random_state(random_state)
        a = min(self.most_frequent,self.num_words)
        # a = self.num_words
        p = np.ones(a)/a
        while True:
            rv = rng.choice(a, size=batch_size, replace=True, p=p)
            yield rv

    def center_embeddings(self):
        mean = self.vectors.mean(0)
        self.vectors = self.vectors-mean

    def normalize(self):
        self.vectors = normalize(self.vectors,'l2')

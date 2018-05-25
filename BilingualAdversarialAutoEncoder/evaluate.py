#!/usr/bin/env python
# encoding: utf-8
import os
import argparse
from collections import OrderedDict
from myembedding import WordEmbeddings
from word_translation import get_word_translation_accuracy
import torch

from timeit import default_timer as timer

torch.cuda.set_device(4)
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=int, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--validation_file", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=int, default=0, help="Normalize embeddings before training")
parser.add_argument("--center_embeddings", type=int, default=0, help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()
'''
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
'''
we1 = WordEmbeddings()
we1.load_from_word2vec_new(params.src_emb)
we2 = WordEmbeddings()
we2.load_from_word2vec_new(params.tgt_emb)

if params.center_embeddings > 0:
    we1.center_embeddings()
    we2.center_embeddings()

if params.normalize_embeddings > 0:
   we1.normalize()
   we2.normalize()

src = torch.from_numpy(we1.vectors).cuda().float()
tgt = torch.from_numpy(we2.vectors).cuda().float()

#print(list(we1.word2id.items())[0])

#exit()
mstart_time = timer()
for method in ["nn","csls_knn_10"]:
    results = get_word_translation_accuracy(
        params.src_lang, we1.word2id, src,
        params.tgt_lang, we2.word2id, tgt,
        method=method,
        #path = params.validation_file
        small_data = False
    )
    acc = results[0][1]
    print("{} to {}".format(params.src_lang,params.tgt_lang))
    print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
    print('Method:{} score:{:.4f}'.format(method,acc))
mstart_time = timer()
for method in ["nn","csls_knn_10"]:
    results = get_word_translation_accuracy(
        params.tgt_lang, we2.word2id, tgt,
        params.src_lang, we1.word2id, src,
        method=method,
        small_data = False
        #path = params.validation_file
    )
    acc = results[0][1]
    print("{} to {}".format(params.tgt_lang,params.src_lang))
    print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
    print('Method:{} score:{:.4f}'.format(method,acc))


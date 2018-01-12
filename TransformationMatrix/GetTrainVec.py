#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: GetTrainVec.py.py
@time: 17-12-18 上午10:11
"""

import numpy as np
import embeddings
import sys
from sklearn import preprocessing


if __name__ == "__main__":

    source_file = open(sys.argv[1], encoding='utf-8', errors='surrogateescape')
    target_file = open(sys.argv[2], encoding='utf-8', errors='surrogateescape')
    en_words,en_vec = embeddings.read(source_file)
    de_words,de_vec = embeddings.read(target_file)

    src_word2ind = {word: i for i, word in enumerate(en_words)}
    trg_word2ind = {word: i for i, word in enumerate(de_words)}

    src_indices = []
    trg_indices = []
    src_words = []
    trg_words = []

    f = open(sys.argv[3], encoding='utf-8', errors='surrogateescape')
    for line in f:
        src, trg = line.split()
        try:
            src_words.append(src)
            src_ind = src_word2ind[src]
            trg_words.append(trg)
            trg_ind = trg_word2ind[trg]

            src_indices.append(src_ind)
            trg_indices.append(trg_ind)
        except KeyError:
            print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

    # origEnVecs=preprocessing.normalize(en_vec)
    # origForeignVecs=preprocessing.normalize(de_vec)

    subsetEnVecs = en_vec[src_indices]
    subsetForeignVecs = de_vec[trg_indices]

    srcfile = open('en.train', mode='w', encoding='utf-8', errors='surrogateescape')
    trgfile = open('de.train', mode='w', encoding='utf-8', errors='surrogateescape')

    embeddings.write(src_words, subsetEnVecs, srcfile)
    embeddings.write(trg_words, subsetForeignVecs, trgfile)
    source_file.close()
    target_file.close()
    srcfile.close()
    trgfile.close()


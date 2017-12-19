#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: dataset.py.py
@time: 17-12-15 下午3:13
"""
import torch
import torch.utils.data             #子类化你的数据
# from torchvision import transforms  #对数据进行预处理
import embeddings
import numpy as np

class MyDataSet(torch.utils.data.Dataset):

    def __init__(self,root,transform=None,train=True):
        self.root = root
        self.train = train
        self.allEmbedding = self.make_dataset(root)


    def __getitem__(self, itemdidx):
        if self.train:
            return self.allEmbedding[itemdidx]

    def make_dataset(self,root):
        root = root.split(' ')
        view1 = open(root[0], encoding='utf-8', errors='surrogateescape')
        view2 = open(root[1], encoding='utf-8', errors='surrogateescape')
        src_words, view1_vec = embeddings.read(view1)
        trg_words, view2_vec = embeddings.read(view2)

        view1.close()
        view2.close()

        return torch.from_numpy(np.column_stack((view1_vec,view2_vec)))

    def __len__(self):
        return len(self.allEmbedding)

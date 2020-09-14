#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : build_word2vec_weights
# @Author   : 张志毅
# @Time     : 2020/9/13 15:34

from itertools import islice

import numpy as np
import torch
from utils.log import logger


def load_word2vec(path=None, word_vocab=None, embedding_dim=None):
    """
    加载词向量
    :param path: None
    :param word_vocab: None
    :param embedding_dim: 768/300
    :return: 返回与word_vocab相对应的vec
    """
    word_vocab_dict = word_vocab.stoi
    vectors_vocab = load_vec(path, embedding_dim=embedding_dim)
    vocab_size = len(word_vocab)
    embed_weights = torch.zeros(vocab_size, embedding_dim)
    for word, index in word_vocab_dict.items():  # 单词和下标
        if word in vectors_vocab:
            em = vectors_vocab[word]
        elif word == '<pad>':
            em = vectors_vocab['[PAD]']
        else:
            em = vectors_vocab['[UNK]']
        embed_weights[index, :] = torch.from_numpy(np.array(em))
    # logger.info("load word2vec weights success...")
    return embed_weights


def load_vec(path=None, embedding_dim=None):
    """
    加载词向量
    :param path: None
    :param embedding_dim: 768/300
    :return: 返回词向量的词典
    """
    vectors_vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in islice(f, 1, None):  # 略过第一行
            items = line.split()
            char, vectors = items[0], items[-embedding_dim:]
            vectors = [float(vector) for vector in vectors]
            vectors_vocab[char] = vectors
    return vectors_vocab
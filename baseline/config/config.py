#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : load_config
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

default_config = {
    'experiment_name': 'test',
    'model_name': 'TransformerEncoderModel',
    'train_path' : '../../data/train_data_all.txt',
    'dev_path' : '../../data/val_data.txt',
    'test_path': '../../data/test1.txt',
    'vocab_path': '../data/vocab/entity_vocab.txt',  # ['vocab/task1_vocab.txt', 'vocab/task1_vocab.val.txt']
    'save_model_path': '../model/save_model/',
    'result_path': '../result/',
    'is_bioes' : False,
    # Baseline Config
    'tag_type': 'BME_SO',  # BIO or BME_SO
    'use_cuda': False,
    'epoch': 100,
    'batch_size': 4,
    'learning_rate': 2e-4,
    'num_layers': 2,
    'pad_index': 1,
    'dropout': 0.5,  # the dropout value
    'embedding_dim': 768,  # embedding dimension     词嵌入: BERT_768 Random_300
    'hidden_dim': 300,
    'use_vectors': True,
    # 'vector_path': 'D:\Python\WorkSpace\zutnlp\\bertvec\\bert_vectors_768.txt',
    'vector_path': '/home/zutnlp/data/bertvec/bert_vectors_768.txt',

    # TransformerEncoder Config
    'n_hid': 200,  # the dimension of the feedforward network model in nn.TransformerEncoder
    'n_layers': 2,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    'n_head': 2,  # the number of heads in the multiheadattention models

    # CNN Config
    'chanel_num': 1,
    'filter_num': 100,

    # Attn Config
    'use_attn': False,
    'key_dim': 64,
    'val_dim': 64,
    'num_heads': 3,
    'attn_dropout': 0.5,

    # FlatLattice Config
    'bi_gram_embed_dim': 150,
    'lattice_embed_dim': 150,

    # RefactorLoss Config
    'use_dae': False,
    'dae_lambda': 1.2,
    'use_dice': False,
    'dice_lambda': 0.01,
}

class Config():
    def __init__(self, **kwargs):
        super(Config, self).__init__()
        for name, value in default_config.items():
            setattr(self, name, value)
    def add_config(self, config_list):
        for name, value in config_list:
            setattr(self, name, value)

config = Config()
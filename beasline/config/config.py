#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : load_config
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

default_config = {
    'train_path' : '../../data/train_data.txt',
    'dev_path' : '../../data/val_data.txt',
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
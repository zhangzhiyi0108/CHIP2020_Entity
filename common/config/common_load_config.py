#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : common_load_config
# @Author   : LiuYan
# @Time     : 2020/8/25 15:57

import torch
import abc
import dynamic_yaml
from base.config.base_load_config import BaseLoadConfig


class CommonLoadConfig(BaseLoadConfig):
    @abc.abstractmethod
    def __init__(self):
        super(CommonLoadConfig, self).__init__()

    def load_config(self, dict_paths: dict):
        _config_path = dict_paths['_config_path']
        with open(_config_path, mode='r') as f:
            configs = dynamic_yaml.load(f)
        configs.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return configs

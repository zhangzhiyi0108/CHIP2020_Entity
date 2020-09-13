#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_module
# @Author   : LiuYan
# @Time     : 2020/8/23 18:34

import abc
import torch.nn as nn


class BaseModule(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        super(BaseModule, self).__init__()

    @abc.abstractmethod
    def forward(self, dict_: dict) -> dict:
        pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_model
# @Author   : LiuYan
# @Time     : 2020/8/23 18:34

import abc
import torch.nn as nn


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def forward(self, dict_inputs: dict) -> dict:
        pass

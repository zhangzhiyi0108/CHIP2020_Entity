#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : common_model
# @Author   : LiuYan
# @Time     : 2020/8/25 19:53

import abc
from base.model.base_model import BaseModel


class CommonModel(BaseModel):
    @abc.abstractmethod
    def __init__(self, config):
        super(CommonModel, self).__init__()
        self.config = config
        self.device = config.device
        # self.rnn = nn.RNN(input_size=3, hidden_size=5)

    @abc.abstractmethod
    def forward(self, dict_inputs: dict) -> dict:
        dict_outputs = dict()
        return dict_outputs

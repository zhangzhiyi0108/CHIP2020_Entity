#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : common_loss
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14
import abc
from torchcrf import CRF
from base.loss.base_loss import BaseLoss


class CommonLoss(BaseLoss):
    @abc.abstractmethod
    def __init__(self, config, data_loader):
        super(CommonLoss, self).__init__()
        self.config = config
        self.device = config.device
        self.data_loader = data_loader

    @abc.abstractmethod
    def forward(self, dict_outputs: dict) -> dict:
        loss_dict = dict()
        loss_dict['generic_loss'] = 100
        return loss_dict

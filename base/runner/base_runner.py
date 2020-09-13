#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_runner
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14

import abc


class BaseRunner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        super(BaseRunner, self).__init__()

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def train_epoch(self):
        pass

    @abc.abstractmethod
    def train_batch(self, dict_batch: dict):
        pass

    @abc.abstractmethod
    def valid_epoch(self):
        pass

    @abc.abstractmethod
    def valid_batch(self, dict_batch: dict):
        pass

    @abc.abstractmethod
    def test(self):
        pass

    @abc.abstractmethod
    def save_model(self, dict_save: dict):
        pass

    @abc.abstractmethod
    def load_model(self, dict_load: dict):
        pass

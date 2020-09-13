#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_data_loader
# @Author   : LiuYan
# @Time     : 2020/8/23 18:34

import abc
from torchtext.data import Dataset


class BaseDataLoader(metaclass=abc.ABCMeta):
    """
    input: conll
    中   B-ORGANIZATION
    共   I-ORGANIZATION
    中   I-ORGANIZATION
    央   E-ORGANIZATION
    致   O
    中   B-ORGANIZATION
    国   I-ORGANIZATION
    致   I-ORGANIZATION
    公   I-ORGANIZATION
    党   I-ORGANIZATION
    十   I-ORGANIZATION
    一   I-ORGANIZATION
    大   E-ORGANIZATION
    output: data_loader(data iter)
    """

    @abc.abstractmethod
    def __init__(self):
        """
        not config
        is train_path and so on
        """
        super(BaseDataLoader, self).__init__()

    @abc.abstractmethod
    def load_data(self, train_path: str, valid_path: str, test_path: str):
        """
        :param train_path: If null, it does not return
        :param valid_path: If null, it does not return
        :param test_path:  If null, it does not return
        :return: train_data, valid_data, test_data
        """
        pass

    @abc.abstractmethod
    def get_iterator(self, *dataset, batch_size=64):
        """
        :param dataset: train_data, valid_data, test_data
        :param batch_size: 64 or other
        :return: train_iter, valid_iter, test_iter
        """
        pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_data_reader
# @Author   : LiuYan
# @Time     : 2020/8/25 12:13

import abc


class BaseDataReader(metaclass=abc.ABCMeta):
    """
    Read the data in the following format:
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
    problem:
    """

    @abc.abstractmethod
    def __init__(self):
        super(BaseDataReader, self).__init__()

    @abc.abstractmethod
    def reade(self, dict_paths: dict):
        pass

    @abc.abstractmethod
    def save(self, dict_paths: dict):
        pass

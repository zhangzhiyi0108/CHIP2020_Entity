#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_data_process
# @Author   : LiuYan
# @Time     : 2020/8/28 10:30

import abc


class BaseDataProcess(metaclass=abc.ABCMeta):
    """
    data processing
    """

    @abc.abstractmethod
    def __init__(self):
        super(BaseDataProcess, self).__init__()

    @abc.abstractmethod
    def process(self, dict_: dict):
        pass

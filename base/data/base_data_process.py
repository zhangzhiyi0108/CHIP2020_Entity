#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_data_process
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14
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

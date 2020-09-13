#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_load_config
# @Author   : LiuYan
# @Time     : 2020/8/25 15:55

import abc


class BaseLoadConfig(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        super(BaseLoadConfig, self).__init__()

    @abc.abstractmethod
    def load_config(self, dict_paths: dict):
        """
        Add the config you need.
        :param dict_paths: *.yml path
        :return: config(YamlDict)
        """
        pass

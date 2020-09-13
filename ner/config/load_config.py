#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : load_config
# @Author   : LiuYan
# @Time     : 2020/8/26 19:15

from common.config.common_load_config import CommonLoadConfig
from utils.log import logger


class LoadConfig(CommonLoadConfig):
    def __init__(self):
        logger.info('Begin load config...')
        super(LoadConfig, self).__init__()


if __name__ == '__main__':
    loadConfig = LoadConfig()
    dict_paths = {'_config_path': '../../base/config/config.yml'}
    config = loadConfig.load_config(dict_paths=dict_paths)
    device = config.device
    pass

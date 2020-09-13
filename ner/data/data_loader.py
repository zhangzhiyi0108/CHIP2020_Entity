#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_loader
# @Author   : LiuYan
# @Time     : 2020/8/25 15:30

from common.data.common_data_loader import CommonDataLoader
from example.config.load_config import LoadConfig
from utils.log import logger


class DataLoader(CommonDataLoader):
    def __init__(self, train_path=None, valid_path=None, test_path=None, device=None, separator='\t', batch_size=64):
        logger.info('Begin load data...')
        super(DataLoader, self).__init__(train_path=train_path, valid_path=valid_path, test_path=test_path,
                                         device=device, separator=separator, batch_size=batch_size)
        logger.info('Finished load data')


if __name__ == '__main__':
    loadConfig = LoadConfig()
    dict_paths = {'_config_path': '../../base/config/config.yml'}
    config = loadConfig.load_config(dict_paths=dict_paths)
    data_loader = DataLoader(train_path=config.data.chip2020entity.train, valid_path=config.data.chip2020entity.valid,
                             test_path=config.data.chip2020entity.test, device=config.device)
    train_iter = data_loader.train_iter
    pass

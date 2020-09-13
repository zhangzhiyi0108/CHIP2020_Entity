#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : common_runner
# @Author   : LiuYan
# @Time     : 2020/8/25 19:53

import abc
import torch
from tqdm import tqdm
import torch.optim as optim
from utils.log import logger
from base.runner.base_runner import BaseRunner


class CommonRunner(BaseRunner):
    @abc.abstractmethod
    def __init__(self, config=None, data_loader=None, model=None, loss=None):
        super(CommonRunner, self).__init__()
        self.loss = loss
        self.model = model
        self.config = config
        self.device = config.device
        self.data_loader = data_loader

    @abc.abstractmethod
    def train(self):
        logger.info('Begin train...')
        for epoch in range(self.config.module.train.epoch):
            total_loss = self.train_epoch()
            p, r, f1 = self.valid_epoch()
            logger.info('epoch: {} generic_loss: {} p: {} r: {} f1: {}'.format(epoch, total_loss['generic_loss'],
                                                                               p, r, f1))
            # the decision to save the model can be made here based on the loss value or f1 value
            self.save_model(save_path='d:/', model_name='model_name')
        logger.info('Finished train')

    @abc.abstractmethod
    def train_epoch(self):
        self.model.train()
        total_loss = {'generic_loss': 0}
        for item in tqdm(self.train_iter):
            dict_batch = {'item': item}
            loss_dict = self.train_batch(dict_batch=dict_batch)
            total_loss['generic_loss'] += loss_dict['generic_loss']
        return total_loss

    @abc.abstractmethod
    def train_batch(self, dict_batch: dict):
        self.optimizer.zero_grad()
        sentence = dict_batch['item'].text[0]
        sent_lengths = dict_batch['item'].text[1]
        targets = dict_batch['item'].tag
        dict_inputs = {'sentence': sentence, 'sent_lengths': sent_lengths}
        dict_outputs = self.model(dict_inputs=dict_inputs)
        dict_outputs = {'outputs': dict_outputs['emissions'], 'targets': targets, 'sentence': sentence}
        loss_dict = self.loss(dict_outputs=dict_outputs)
        loss_dict['generic_loss'].backward()
        self.optimizer.step()
        return loss_dict

    @abc.abstractmethod
    def valid_epoch(self):
        self.model.eval()
        p, r, f1 = 0.8, 0.8, 0.8
        for item in tqdm(self.valid_iter):
            dict_batch = {'item': item}
            self.valid_batch(dict_batch=dict_batch)
        return p, r, f1

    @abc.abstractmethod
    def valid_batch(self, dict_batch: dict):
        sentence = dict_batch['item'].text[0]
        sent_lengths = dict_batch['item'].text[1]
        dict_inputs = {'sentence': sentence, 'sent_lengths': sent_lengths}
        dict_outputs = self.model(dict_inputs=dict_inputs)
        dict_outputs['targets'] = dict_batch['item'].tag
        # begin valid.
        pass

    @abc.abstractmethod
    def test(self):
        logger.info('Begin test...')
        logger.info('Finished test')

    def save_model(self, dict_save: dict):
        save_path = dict_save['save_path']
        model_name = dict_save['model_name']
        torch.save(self.model, save_path + '/' + model_name + '.pkl')

    def load_model(self, dict_load: dict):
        load_path = dict_load['load_path']
        model_name = dict_load['model_name']
        self.model = torch.load(load_path + '/' + model_name + '.pkl')

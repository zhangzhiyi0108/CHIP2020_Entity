#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_bilstm_crf
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14

import torch
import warnings
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import classification_report
from common.runner.common_runner import CommonRunner
from example.config.load_config import LoadConfig
from example.data.data_loader import DataLoader
from example.model.bilstm_crf import BiLSTMCRF
from example.loss.loss import Loss
from utils.log import logger

warnings.filterwarnings('ignore')


class Runner(CommonRunner):
    def __init__(self, config=None, data_loader=None, model=None, loss=None):
        super(Runner, self).__init__(config=config, data_loader=data_loader, model=model, loss=loss)
        self.word_vocab = data_loader.word_vocab
        self.tag_vocab = data_loader.tag_vocab
        self.train_iter = data_loader.train_iter
        self.valid_iter = data_loader.valid_iter
        self.optimizer = optim.Adam(model.parameters(), lr=config.module.train.lr, weight_decay=1e-5)

    def train(self):
        logger.info('Begin train...')
        dict_save = {'save_path': 'd:', 'model_name': 'bilstm_crf'}
        for epoch in range(self.config.module.train.epoch):
            total_loss = self.train_epoch()
            dict_score = self.valid_epoch()
            weighted_avg = dict_score['weighted avg']
            logger.info('epoch: {} crf_loss: {}'.format(epoch + 1, total_loss['crf_loss']))
            logger.info('{}'.format(weighted_avg))
            # the decision to save the model can be made here based on the loss value or f1 value
            self.save_model(dict_save=dict_save)
        logger.info('Finished train.')

    def train_epoch(self):
        self.model.train()
        total_loss = {'crf_loss': 0}
        for item in tqdm(self.train_iter):
            dict_batch = {'item': item}
            loss_dict = self.train_batch(dict_batch=dict_batch)
            total_loss['crf_loss'] += loss_dict['crf_loss']
        return total_loss

    def train_batch(self, dict_batch: dict):
        self.optimizer.zero_grad()
        sentence = dict_batch['item'].text[0]
        sent_lengths = dict_batch['item'].text[1]
        targets = dict_batch['item'].tag
        dict_inputs = {'sentence': sentence, 'sent_lengths': sent_lengths}
        dict_outputs = self.model(dict_inputs=dict_inputs)
        dict_outputs = {'outputs': dict_outputs['emissions'], 'targets': targets, 'sentence': sentence}
        loss_dict = self.loss(dict_outputs=dict_outputs)
        loss_dict['crf_loss'].backward()
        self.optimizer.step()
        return loss_dict

    def valid_epoch(self):
        self.model.eval()
        tag_true_all = []
        tag_pred_all = []
        for item in tqdm(self.valid_iter):
            dict_batch = {'item': item}
            tag_true_batch, tag_pred_batch = self.valid_batch(dict_batch=dict_batch)
            tag_true_all.extend(tag_true_batch)
            tag_pred_all.extend(tag_pred_batch)
        labels = []
        for index, label in enumerate(self.tag_vocab.itos):
            labels.append(label)
        labels.remove('O')
        dict_score = classification_report(tag_true_all, tag_pred_all, labels=labels, output_dict=True)
        return dict_score

    def valid_batch(self, dict_batch: dict):
        tag_true_batch = []
        tag_pred_batch = []
        sentence = dict_batch['item'].text[0]
        sent_lengths = dict_batch['item'].text[1]
        targets = torch.transpose(dict_batch['item'].tag, 0, 1)
        tag = targets.to('cpu').numpy().tolist()
        dict_inputs = {'sentence': sentence, 'sent_lengths': sent_lengths}
        dict_outputs = self.model(dict_inputs=dict_inputs)
        result = dict_outputs['outputs']
        assert len(tag) == len(result), 'tag_len: {} != result_len: {}'.format(len(tag), len(result))
        for i, result_list in zip(range(sentence.size(1)), result):
            tag_list = tag[i][:sent_lengths[i]]
            assert len(tag_list) == len(result_list), 'tag_list: {} != result_list: {}'.format(len(tag_list),
                                                                                               len(result_list))
            tag_true = [self.tag_vocab.itos[k] for k in tag_list]
            tag_true_batch.extend(tag_true)
            tag_pred = [self.tag_vocab.itos[k] for k in result_list]
            tag_pred_batch.extend(tag_pred)
        return tag_true_batch, tag_pred_batch

    def test(self):
        logger.info('Begin test...')
        logger.info('Finished test.')


if __name__ == '__main__':
    loadConfig = LoadConfig()
    dict_paths = {'_config_path': '../../base/config/config.yml'}
    config = loadConfig.load_config(dict_paths=dict_paths)
    data_loader = DataLoader(train_path=config.data.chip2020entity.train, valid_path=config.data.chip2020entity.valid,
                             test_path=config.data.chip2020entity.test, device=config.device,
                             batch_size=config.module.train.batch_size)
    model = BiLSTMCRF(config=config, data_loader=data_loader).to(config.device)
    loss = Loss(config=config, data_loader=data_loader)
    runner = Runner(config=config, data_loader=data_loader, model=model, loss=loss)
    runner.train()

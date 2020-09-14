#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : module
# @Author   : 张志毅
# @Time     : 2020/9/13 15:39

import codecs
import os
import sys
import json
import warnings
import numpy
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from baseline.config.config import config, DEVICE
from baseline.data.data_loader import tool
from baseline.model.transformer_bilstm_crf import TransformerEncoderModel
from utils.log import logger

warnings.filterwarnings('ignore')


class CHIP2020_NER():
    def __init__(self):
        self.config = config
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.unlabeled_data = None
        self.word_vocab = None
        self.tag_vocab = None
        self.bi_gram_vocab = None
        self.lattice_vocab = None
        self.train_iter = None
        self.dev_iter = None
        self.test_iter = None
        self.unlabeled_iter = None
        self.model_name = config.model_name
        # self.experiment_name = config.experiment_name

    def init_model(self, config=None, word_vocab=None, vocab_size=None, tag_num=None, vectors_path=None):
        model_name = config.model_name
        models = {
            'TransformerEncoderModel': TransformerEncoderModel,
        }
        model = models[model_name](config, word_vocab, vocab_size, tag_num, vectors_path).to(DEVICE)
        return model

    def train(self):
        logger.info('Loading data ...')
        self.train_data = tool.load_data(config.train_path, config.is_bioes)
        self.dev_data = tool.load_data(config.dev_path, config.is_bioes)

        self.word_vocab = tool.get_text_vocab(self.train_data, self.dev_data)
        self.tag_vocab = tool.get_tag_vocab(self.train_data, self.dev_data)
        self.train_iter = tool.get_iterator(self.train_data, batch_size=config.batch_size)
        self.dev_iter = tool.get_iterator(self.dev_data, batch_size=config.batch_size)

        # if self.model_name == 'TransformerEncoderModel':
        #         #     self.model = TransformerEncoderModel(config, self.word_vocab,len(self.word_vocab), len(self.tag_vocab),config.vector_path).to(DEVICE)
        model = self.init_model(self.config, self.word_vocab, len(self.word_vocab), len(self.tag_vocab),
                                config.vector_path)
        self.model = model
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        p_max = 0
        r_max = 0
        f1_max = 0
        best_epoch = -1
        logger.info('Beginning train ...')

        for epoch in range(config.epoch):
            model.train()
            acc_loss = 0
            # print(optimizer.param_groups[0]['lr'])
            # tool.adjust_learning_rate(optimizer, epoch + 1)
            for item in tqdm(self.train_iter):
                optimizer.zero_grad()
                tag = item.tag
                text = item.text[0]
                text_len = item.text[1]
                loss = self.model.loss(text, text_len, tag)
                # refactor_loss = loss['refactor_loss']
                # total_loss = self.acc_total_loss(total_loss, loss)
                acc_loss += loss.view(-1).cpu().data.tolist()[0]
                loss.backward()
                optimizer.step()

            p, r, f1 = self.evaluate()
            logger.info('precision: {:.4f} recall: {:.4f} f1: {:.4f} loss: {}'.format(p, r, f1, acc_loss))

            # scheduler.step(f1)
            if f1 > f1_max:
                p_max = p
                r_max = r
                f1_max = f1
                best_epoch = epoch + 1
                logger.info('save best model...')
                torch.save(self.model.state_dict(),
                           config.save_model_path + 'model_{}.pkl'.format(self.experiment_name))
                logger.info(
                    'best model: precision: {:.4f} recall: {:.4f} f1: {:.4f} epoch: {}'.format(p_max, r_max, f1_max,
                                                                                               best_epoch))

    def evaluate(self):
        self.model.eval()
        tag_true_all = []
        tag_pred_all = []

        model = self.model
        for index, iter in enumerate(tqdm(self.dev_iter)):
            if iter.tag.shape[1] == self.config.batch_size:

                text = iter.text[0]
                tag = torch.transpose(iter.tag, 0, 1)
                text_len = iter.text[1]
                result = model(text, text_len)
                for i, result_list in zip(range(text.size(1)), result):
                    text1 = text.permute(1, 0)
                    sentence = [self.word_vocab.itos[w] for w in text1[i][:text_len[i]]]
                    tag_list = tag[i][:text_len[i]]
                    assert len(tag_list) == len(result_list), 'tag_list: {} != result_list: {}'.format(
                        len(tag_list), len(result_list))
                    tag_true = [self.tag_vocab.itos[k] for k in tag_list]
                    tag_true_all.extend(tag_true)
                    tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                    tag_pred_all.extend(tag_pred)
        labels = []
        for index, label in enumerate(self.tag_vocab.itos):
            labels.append(label)
        labels.remove('O')
        p, r, f1 = classification_report(tag_true_all, tag_pred_all, labels=labels)
        return p, r, f1

    def acc_total_loss(self, total_loss=None, loss=None):
        for loss_name in loss:
            if loss[loss_name] is not None:
                total_loss[loss_name] += loss[loss_name].view(-1).cpu().data.tolist()[0]
        return total_loss


if __name__ == '__main__':
    CHIP2020_NER = CHIP2020_NER()
    CHIP2020_NER.train()

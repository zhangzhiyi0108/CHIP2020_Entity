#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : common_data_loader
# @Author   : LiuYan
# @Time     : 2020/8/25 15:29

import abc
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from base.data.base_data_loader import BaseDataLoader


def tokenizer(token):
    return [k for k in token]


TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer, include_lengths=True)
TAG = Field(sequential=True, use_vocab=True, tokenize=tokenizer, is_target=True, pad_token=None)
fields = [('text', TEXT), ('tag', TAG)]


class CommonDataLoader(BaseDataLoader):
    @abc.abstractmethod
    def __init__(self, train_path: str, valid_path: str, test_path: str, device: str, separator='\t', batch_size=64):
        super(CommonDataLoader, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.word_vocab = None
        self.tag_vocab = None
        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None
        self.load_data(train_path=train_path, valid_path=valid_path, test_path=test_path, separator=separator)

    def load_data(self, train_path: str, valid_path: str, test_path: str, separator='\t'):
        self.train_data = SequenceTaggingDataset(path=train_path, fields=fields, separator=separator)
        self.valid_data = SequenceTaggingDataset(path=valid_path, fields=fields, separator=separator)
        self.test_data = SequenceTaggingDataset(path=test_path, fields=fields, separator=separator)
        self.get_vocab(self.train_data, self.valid_data, self.test_data)
        self.get_iterator(self.train_data, self.valid_data, self.test_data)

    def get_vocab(self, *dataset):
        """
        :param dataset: train_data, valid_data, test_data
        :return: text_vocab, tag_vocab
        """
        TEXT.build_vocab(*dataset)
        TAG.build_vocab(*dataset)
        self.word_vocab = TEXT.vocab
        self.tag_vocab = TAG.vocab

    def get_iterator(self, *dataset):
        self.train_iter = BucketIterator(dataset[0], batch_size=self.batch_size, shuffle=False,
                                         sort_key=lambda x: len(x.text), sort_within_batch=True, device=self.device)
        self.valid_iter = BucketIterator(dataset[1], batch_size=self.batch_size, shuffle=False,
                                         sort_key=lambda x: len(x.text), sort_within_batch=True, device=self.device)
        self.test_iter = BucketIterator(dataset[2], batch_size=self.batch_size, shuffle=False,
                                        sort_key=lambda x: len(x.text), sort_within_batch=True, device=self.device)

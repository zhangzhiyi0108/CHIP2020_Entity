#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_loader
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14
from beasline.config.config import config, device
from utils.log import logger
from torchtext.data import Field, BucketIterator, Example, Dataset


def tokenizer(token):
    return [k for k in token]


TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer, include_lengths=True)
TAG = Field(sequential=True, use_vocab=True, tokenize=tokenizer, is_target=True, pad_token=None)
Fields = [('text', TEXT), ('tag', TAG)]


class DataLoader(Dataset):
    def __init__(self, path, fields, **kwargs):
        examples = []
        self.encoding = 'utf-8'
        with open(path, 'r', encoding='utf -8') as f:
            lines = f.readlines()
            for example in lines:
                # self.fields = fields
                self.example = self.get_list(example)
                examples.append(Example.fromlist(self.example, fields))
        super(DataLoader, self).__init__(examples, fields, **kwargs)

    def get_list(self, example):

        example = example.split('|||', 1)
        text = example[0]
        tag_list = self.get_tag(example)
        text_list = [x for x in text]
        assert len(text_list) == len(tag_list)
        return text_list, tag_list

    def get_tag(self, example):
        text = example[0]
        # text = text + '。'
        entities = example[1].split('|||')
        tag_list = ['O' for i in range(len(text))]
        while '' in entities:
            entities.remove('')
        while '\n' in entities:
            entities.remove('\n')
        for entity in entities:
            entity = entity.split('    ')
            start_pos = int(entity[0])
            end_pos = int(entity[1])
            label_type = entity[2]
            tag_list[start_pos] = 'B-' + str(label_type)
            for i in range(start_pos + 1, end_pos):
                tag_list[i] = 'I-' + str(label_type)
        return tag_list


class Tool():
    def __init__(self):
        if config is not None:
            self.fields = Fields

    def load_data(self, path: str):
        fields = self.fields
        dataset = DataLoader(path, fields=fields)
        return dataset

    def get_text_vocab(self, *dataset):
        TEXT.build_vocab(*dataset)
        return TEXT.vocab

    def get_tag_vocab(self, *dataset):
        TAG.build_vocab(*dataset)
        return TAG.vocab

    def get_iterator(self, dataset: Dataset, batch_size=16,
                     sort_key=lambda x: len(x.text), sort_within_batch=True):
        iterator = BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                                  sort_within_batch=sort_within_batch, device=device)
        return iterator


tool = Tool()
if __name__ == '__main__':
    train_data = tool.load_data(config.train_path)
    dev_data = tool.load_data(config.dev_path)
    text_vocab = tool.get_text_vocab(train_data, dev_data)
    tag_vocab = tool.get_tag_vocab(train_data, dev_data)
    train_iter = tool.get_iterator(train_data, batch_size=16)
    pass

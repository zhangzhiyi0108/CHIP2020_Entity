#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : bilstm_crf
# @Author   : LiuYan
# @Time     : 2020/8/26 19:15

import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common.model.common_model import CommonModel
from utils.log import logger


class BiLSTMCRF(CommonModel):
    def __init__(self, config, data_loader):
        logger.info('Begin create model...')
        super(BiLSTMCRF, self).__init__(config=config)
        self.embedding_dim = config.model.bilstm_crf.embedding_dim
        self.hidden_dim = config.model.bilstm_crf.hidden_dim
        self.num_layers = config.model.bilstm_crf.num_layers
        self.dropout = config.model.bilstm_crf.dropout
        self.vocab_size = len(data_loader.word_vocab)
        self.tag_num = len(data_loader.tag_vocab)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.device)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                            bidirectional=True, num_layers=self.num_layers, dropout=self.dropout).to(self.device)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num).to(self.device)
        self.crf = CRF(self.tag_num)
        logger.info('Finished create model')

    def init_hidden(self, batch_size=None):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(self.device)
        return h0, c0

    def forward(self, dict_inputs: dict) -> dict:
        """
        :param dict_inputs: {sentence, sent_lengths}
        :return:
        """
        dict_outputs = dict()
        sentence = dict_inputs['sentence']
        sent_lengths = dict_inputs['sent_lengths']
        mask = torch.ne(sentence, 1)
        input_embed = self.embedding(sentence.to(self.device)).to(self.device)
        input = pack_padded_sequence(input=input_embed, lengths=sent_lengths)
        hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, _ = self.lstm(input, hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(self.device))
        dict_outputs['emissions'] = self.hidden2label(lstm_out.to(self.device)).to(self.device)
        dict_outputs['outputs'] = self.crf.decode(emissions=dict_outputs['emissions'], mask=mask)
        return dict_outputs

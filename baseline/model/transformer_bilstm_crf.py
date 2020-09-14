#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : bilstm_crf
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14
import math
import numpy
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.nn import TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from baseline.config.config import DEVICE
from baseline.loss.loss import DiceLoss1
from utils.build_word2vec_weights import load_word2vec
from utils.log import logger


class TransformerEncoderModel(nn.Module):
    def __init__(self, config, word_vocab, vocab_size, tag_num, vector_path):
        super(TransformerEncoderModel, self).__init__()
        self.use_dae = config.use_dae
        self.dae_lambda = config.dae_lambda
        self.use_dice = config.use_dice
        self.dice_lambda = config.dice_lambda
        self.vocab_size = vocab_size
        self.word_vocab = word_vocab
        self.tag_num = tag_num
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.bidirectional = True
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.drop = nn.Dropout(self.dropout)
        self.vector_path = vector_path
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.embedding_dim, self.dropout)
        encoder_layers = TransformerEncoderLayer(self.embedding_dim, config.n_head, config.n_hid, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.n_layers)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if config.use_vectors:
            logger.info('Loading word vectors from {}...'.format(self.vector_path))
            embed_weights = load_word2vec(self.vector_path, self.word_vocab, embedding_dim=self.embedding_dim)
            logger.info('Finished load word vectors')
            self.embedding = nn.Embedding.from_pretrained(embed_weights, freeze=False).to(DEVICE)
        # self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                            bidirectional=self.bidirectional, num_layers=1).to(DEVICE)
        self.linear = nn.Linear(self.hidden_dim, self.tag_num)
        self.lm_decoder = nn.Linear(self.hidden_dim, self.vocab_size)
        self.init_weights()
        self.crf_layer = CRF(self.tag_num)
        self.dice_loss = DiceLoss1()
        self.criterion = nn.CrossEntropyLoss()


    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def init_hidden_lstm(self):
        return (torch.randn(2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(DEVICE),
                torch.randn(2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(DEVICE))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _get_src_key_padding_mask(self, text_len, seq_len):
        batch_size = text_len.size(0)
        list1 = []
        for i in range(batch_size):
            list2 = []
            list2.append([False for i in range(text_len[i])] + [True for i in range(seq_len - text_len[i])])
            list1.append(list2)
        src_key_padding_mask = torch.tensor(numpy.array(list1)).squeeze(1)
        return src_key_padding_mask

    def loss(self, src, text_len, tag):
        loss = {'crf_loss': None,
                'dae_loss': None,
                'dice_loss': None,
                'refactor_loss': None}
        mask_crf = torch.ne(src, 1)
        transformer_out = self.transformer_forward(src, text_len)
        lstm_out, _ = self.lstm(transformer_out)
        emissions = self.linear(lstm_out)
        crf_loss = -self.crf_layer(emissions, tag, mask=mask_crf)
        return crf_loss

    def dae_loss(self, src, text_len):
        src_encoding = self.encode(src, text_len)
        lm_output = self.decode_lm(src_encoding)
        lm_loss = self.criterion(lm_output.view(-1, self.vocab_size), src.view(-1))
        return lm_loss

    def dae_forward(self, src, text_len):
        pass

    def forward(self, src, text_len):
        mask_crf = torch.ne(src, 1)
        transformer_out = self.transformer_forward(src, text_len)
        lstm_out, _ = self.lstm(transformer_out)
        # att_out = torch.bmm(lstm_out.transpose(0,1), self.att_weight.transpose(0,1)).transpose(0,1)
        emissions = self.linear(lstm_out)
        return self.crf_layer.decode(emissions, mask=mask_crf)

    def transformer_forward(self, src, text_len):
        src_key_padding_mask = self._get_src_key_padding_mask(text_len, src.size(0))
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask.to(DEVICE))
        return output

    def _encode(self, source):
        _, hidden = self.lstm(source)
        output, _ = self.lstm(source, hidden)
        return output

    def encode(self, source, length):
        # _, hidden = self.lstm(source)
        # output, _ = self.lstm(source, hidden)
        embed = self.embedding(source)
        packed_src_embed = pack_padded_sequence(embed, length)
        _, hidden = self.lstm(packed_src_embed)
        embed = self.drop(self.embedding(source))
        packed_src_embed = pack_padded_sequence(embed, length)
        lstm_output, _ = self.lstm(packed_src_embed, hidden)
        lstm_output = pad_packed_sequence(lstm_output)
        lstm_output = self.drop(lstm_output[0])
        return lstm_output

    def decode_lm(self, src_encoding):
        decoded = self.lm_decoder(
            src_encoding.contiguous().view(src_encoding.size(0) * src_encoding.size(1), src_encoding.size(2)))
        lm_output = decoded.view(src_encoding.size(0), src_encoding.size(1), decoded.size(1))
        return lm_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
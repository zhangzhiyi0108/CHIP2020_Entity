#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : module
# @Author   : 张志毅
# @Time     : 2020/9/13 15:39

import warnings

import numpy
import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm

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
        self.experiment_name = config.experiment_name

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
        model = self.init_model(self.config, self.word_vocab, len(self.word_vocab), len(self.tag_vocab),
                                config.vector_path)
        self.model = model
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        f1_max = 0
        p_max = 0
        r_max = 0
        logger.info('Beginning train ...')
        for epoch in range(config.epoch):
            model.train()
            acc_loss = 0
            for item in tqdm(self.train_iter):
                optimizer.zero_grad()
                tag = item.tag
                text = item.text[0]
                text_len = item.text[1]
                loss = self.model.loss(text, text_len, tag)
                acc_loss += loss.view(-1).cpu().data.tolist()[0]
                loss.backward()
                optimizer.step()

            entity_prf_dict, prf_dict = self.evaluate()
            lable_report = prf_dict['weighted avg']
            entity_report = entity_prf_dict['average']
            f1 = entity_report['f1-score']
            p = entity_report['precision']
            r = entity_report['recall']
            logger.info('epoch: {} precision: {:.4f} recall: {:.4f} f1: {:.4f} loss: {}'.format(epoch, p, r, f1, acc_loss))
            if f1 > f1_max:
                f1_max = f1
                p_max = p
                r_max = r
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
        entity_prf_dict={}
        # S : 预测输出的结果
        # G ：人工标注的正确的结果
        entities_total = {'疾病(dis)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '临床表现(sym)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '医疗程序(pro)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '医疗设备(equ)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '药物(dru)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '医学检验项目(ite)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '身体(bod)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '科室(dep)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '微生物类(mic)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0}
                          }
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
                    entities = self._evaluate(sentence=sentence, tag_true=tag_true, tag_pred=tag_pred)
                    assert len(entities_total) == len(entities), 'entities_total: {} != entities: {}'.format(
                        len(entities_total), len(entities))
                    for entity in entities_total:
                        entities_total[entity]['TP'] += entities[entity]['TP']
                        entities_total[entity]['S'] += entities[entity]['S']
                        entities_total[entity]['G'] += entities[entity]['G']
        TP = 0
        S = 0
        G = 0
        print('\n--------------------------------------------------')
        print('\tp\t\t\tr\t\t\tf1\t\t\tlabel_type')
        for entity in entities_total:
            entities_total[entity]['p'] = entities_total[entity]['TP'] / entities_total[entity]['S'] \
                if entities_total[entity]['S'] != 0 else 0
            entities_total[entity]['r'] = entities_total[entity]['TP'] / entities_total[entity]['G'] \
                if entities_total[entity]['G'] != 0 else 0
            entities_total[entity]['f1'] = 2 * entities_total[entity]['p'] * entities_total[entity]['r'] / (
                    entities_total[entity]['p'] + entities_total[entity]['r']) \
                if entities_total[entity]['p'] + entities_total[entity]['r'] != 0 else 0
            print('\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{}'.format(entities_total[entity]['p'], entities_total[entity]['r'],
                                                              entities_total[entity]['f1'], entity))
            entity_dict = {'precision': entities_total[entity]['p'], 'recall': entities_total[entity]['r'],
                           'f1-score': entities_total[entity]['f1'], 'support': ''}
            entity_prf_dict[entity] = entity_dict
            TP += entities_total[entity]['TP']
            S += entities_total[entity]['S']
            G += entities_total[entity]['G']
        p = TP / S if S != 0 else 0
        r = TP / G if G != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        print('\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\taverage'.format(p, r, f1))
        print('--------------------------------------------------')
        entity_prf_dict['average'] = {'precision': p, 'recall': r, 'f1-score': f1, 'support': ''}

        labels = []
        for index, label in enumerate(self.tag_vocab.itos):
            labels.append(label)
        labels.remove('O')
        prf_dict = classification_report(tag_true_all, tag_pred_all, labels=labels, output_dict=True)
        # print(classification_report(tag_true_all, tag_pred_all, labels=labels))
        return entity_prf_dict, prf_dict

    def _evaluate(self, sentence=None, tag_true=None, tag_pred=None):
        """
        先对true进行还原成 [{}] 再对pred进行还原成 [{}]
        :param tag_true: list[]
        :param tag_pred: list[]
        :return:
        """
        true_list = self._build_list_dict(len(tag_true), tag_true, sentence)
        pred_list = self._build_list_dict(len(tag_pred), tag_pred, sentence)
        entities = {      '疾病(dis)': {'TP': 0, 'S': 0, 'G': 0},
                          '临床表现(sym)': {'TP': 0, 'S': 0, 'G': 0},
                          '医疗程序(pro)': {'TP': 0, 'S': 0, 'G': 0},
                          '医疗设备(equ)': {'TP': 0, 'S': 0, 'G': 0},
                          '药物(dru)': {'TP': 0, 'S': 0, 'G': 0},
                          '医学检验项目(ite)': {'TP': 0, 'S': 0, 'G': 0},
                          '身体(bod)': {'TP': 0, 'S': 0, 'G': 0},
                          '科室(dep)': {'TP': 0, 'S': 0, 'G': 0},
                          '微生物类(mic)': {'TP': 0, 'S': 0, 'G': 0}
                          }
        for true in true_list:
            label_type = true['label_type']
            entities[label_type]['G'] += 1
        for pred in pred_list:
            label_type = pred['label_type']
            label_name = pred['name']
            label_start = pred['start_pos']
            label_end = pred['end_pos']
            entities[label_type]['S'] += 1
            for true in true_list:
                if label_type == true['label_type'] and label_name == true['name'] and label_start == true[
                    'start_pos'] and label_end == true['end_pos']:
                    entities[label_type]['TP'] += 1
        # self.record_pred_info(sentence=sentence, true_list=true_list, pred_list=pred_list,
        #                       path='./result/classification_report/{}/pred_info.txt'.format(config.experiment_name))
        return entities

    def _build_list_dict(self, _len, _list, sentence):
        build_list = []
        tag_dict = {      'dis':'疾病(dis)',
                          'sym':'临床表现(sym)',
                          'pro':'医疗程序(pro)',
                          'equ':'医疗设备(equ)',
                          'dru':'药物(dru)' ,
                          'ite':'医学检验项目(ite)' ,
                          'bod':'身体(bod)',
                          'dep':'科室(dep)',
                          'mic':'微生物类(mic)'
                    }
        i = 0

        while i<_len:
            if _list[i][0] == 'B':
                label_type = _list[i][2:]
                start_pos = i
                end_pos = start_pos
                if end_pos != _len-1:
                    if end_pos+1<_len and _list[end_pos+1][0] != 'I':
                        end_pos+=1
                    else:
                        while end_pos+1 < _len and _list[end_pos+1][0] == 'I' and _list[end_pos+1][2:] == label_type:
                            end_pos += 1
                build_list.append({'name': ''.join(sentence[start_pos:end_pos+1]), 'start_pos':start_pos, 'end_pos':end_pos, 'label_type': tag_dict[label_type]})
                i = end_pos+1
            else:
                i+=1
        result = []
        for dict1 in build_list:
            if dict1 not in result:
                result.append(dict1)
        return result


    def predict(self, path=None, model_name=None, save_path=None):
        if path is None:
            path = config.test_path
            model_name = self.config.save_model_path + 'model_{}.pkl'.format(self.config.experiment_name)
            save_path = self.config.result_path + 'result_{}.txt'.format(self.config.experiment_name)
        train_data = tool.load_data(config.train_path, config.is_bioes)
        dev_data = tool.load_data(config.dev_path, config.is_bioes)
        logger.info('Finished load data...')
        word_vocab = tool.get_text_vocab(train_data, dev_data)
        tag_vocab = tool.get_tag_vocab(train_data, dev_data)
        logger.info('Finished build vocab...')

        model = self.init_model(self.config, word_vocab, len(word_vocab), len(tag_vocab), config.vector_path)
        model.load_state_dict(torch.load(model_name))
        with open(path, 'r', encoding='utf -8') as f:
            with open(save_path, 'w', encoding='utf-8') as fw:
                lines = f.readlines()
                for line in tqdm(lines):
                    text = torch.tensor(numpy.array([word_vocab.stoi[word] for word in line], dtype='int64')).unsqueeze(
                        1).expand(len(line), self.config.batch_size).to(DEVICE)
                    text_len = torch.tensor(numpy.array([len(line)], dtype='int64')).expand(self.config.batch_size).to(
                        DEVICE)
                    result = model(text, text_len)[0]
                    tag_pred = [tag_vocab.itos[k] for k in result]
                    sentence = line.replace('\n', '')
                    result_line = self._bulid_result_line(sentence, tag_pred)
                    fw.write(result_line + '\n')
            fw.close()
        f.close()

    def _bulid_result_line(self, sentence, tag_pred):
        result_list = []
        for index, tag in zip(range(len(tag_pred)), tag_pred):
            if tag[0] == 'B':
                start = index
                end = index
                label_type = tag[2:]
                if end != len(tag_pred) - 1:
                    while tag_pred[end + 1][0] == 'I' and tag_pred[end + 1][2:] == label_type:
                        end += 1
                result_list.append({'start': start,
                                    'end': end,
                                    'lable_type': label_type

                                    })
        line = ''.join(sentence)
        if len(result_list) != 0:
            for index, item in enumerate(result_list):
                line = line + '|||' + str(result_list[index]['start']) + '    ' + str(
                    result_list[index]['end']) + '    ' + str(result_list[index]['lable_type'])
            line = line + '|||'
        else:
            line = line
        return line

if __name__ == '__main__':
    CHIP2020_NER = CHIP2020_NER()
    CHIP2020_NER.train()
    CHIP2020_NER.predict()

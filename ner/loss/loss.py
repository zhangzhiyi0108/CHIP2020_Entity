#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : loss
# @Author   : LiuYan
# @Time     : 2020/8/26 19:14

import torch
from torchcrf import CRF
from common.loss.common_loss import CommonLoss
from utils.log import logger


class Loss(CommonLoss):
    def __init__(self, config, data_loader):
        logger.info('Begin create loss...')
        super(Loss, self).__init__(config=config, data_loader=data_loader)
        self.tag_num = len(data_loader.tag_vocab)
        self.crf = CRF(self.tag_num).to(self.device)
        logger.info('Finished create loss')

    def forward(self, dict_outputs: dict) -> dict:
        """
        :param dict_outputs: {outputs, targets, sentence}
        :return: loss_dict: example {crf_loss, dae_loss, dice_loss, refactor_loss}
        """
        outputs = dict_outputs['outputs']
        targets = dict_outputs['targets']
        sentence = dict_outputs['sentence']
        loss_dict = dict()
        mask_crf = torch.ne(sentence, 1)
        loss_dict['crf_loss'] = -self.crf(outputs, targets, mask=mask_crf) / targets.size(1)
        return loss_dict

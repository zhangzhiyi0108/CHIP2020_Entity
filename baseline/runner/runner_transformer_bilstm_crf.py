#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_bilstm_crf
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14

import torch
import warnings

from baseline.module.module import CHIP2020_NER
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    CHIP2020_NER = CHIP2020_NER()
    CHIP2020_NER.train()
    CHIP2020_NER.predict()

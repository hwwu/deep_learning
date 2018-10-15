#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/20 下午3:25
# @Author   :hwwu
# @File     :split_word.py

path = '/Users/liyangyang/Downloads/bdci/'

import pandas as pd, numpy as np

train = pd.read_csv(path + 'train.csv')
print(train.shape)
stopword_path = '/Users/liyangyang/Downloads/stopwords/stopwords1893.txt'
import jieba
import fool
def stopwordslist():
    stopwords = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]
    # stopwords = ['，', '。', '、', '...', '“', '”', '《', '》', '：', '；']
    return stopwords

import codecs
f = codecs.open(path+'train_no_lable.txt', 'a', 'utf8')
train_doc_list = []
for i in range(100):
    print(train['content'][i].strip())
    print('..........')
    sentence_seged = jieba.cut(train['content'][i].strip())
    outstr = ''
    for word in sentence_seged:
        if (word != '\t') & (word.strip() != ''):
            outstr += word
            outstr += ' '
    print(outstr)
    print('..........')

    sentence_seged_fool = fool.cut(train['content'][i].strip())
    print(sentence_seged_fool)
    print('***********')

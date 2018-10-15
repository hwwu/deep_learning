#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/5 下午2:15
# @Author   :hwwu
# @File     :par.py

import pandas as pd, numpy as np

# path = '/Users/liyangyang/Downloads/dwb/new_data/'
path = '/Users/liyangyang/Downloads/bdci/'

# train = pd.read_csv(path + 'train_set.csv')['word_seg']

from gensim.models import word2vec

# train = np.load(path + 'vocab/vocab_train.npy')
# test = np.load(path + 'vocab/vocab_test.npy')
#
# test = np.array(test)
#
# total = np.append(train, test, axis=0)
#
# t =[]
# for i in range(len(train)):
#     row = train[i]
#     r = []
#     for j in range(len(row)):
#         r.append(str(row[j]))
#     r = np.reshape(r,[len(row),1])
#     t.append(r)
# t = np.array(t)
# print(train.shape)
#
# sentences = word2vec.PathLineSentences(path+'train_no_lable.txt')
# model = word2vec.Word2Vec(sentences,size=128, min_count=1, iter=10,workers=10)
#
# model.save(path+'word2vec/model')
#
#
model = word2vec.Word2Vec.load(path+'word2vec/model')
#
# print(train[0])
print(model.wv['系统'])

similarities=model.wv.most_similar('系统',topn=20)

for word , score in similarities:
    print(word , score)

# y1 = model.similarity('1', '2')
# print(y1)
#
# y2 = model.similarity('1', '3')
# print(y2)

# word2vec.word2vec('/Users/liyangyang/Downloads/word2vec-0.10.2/README.md',path+'word2vec/model.bin',size=128,iter_=10,threads=10,min_count=1)

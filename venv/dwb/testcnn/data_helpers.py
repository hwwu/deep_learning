#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/8/20 下午4:16
# @Author   :hwwu
# @File     :data_helpers.py

import numpy as np
import pandas as pd
path = '/Users/liyangyang/Downloads/dwb/new_data/'
column = "word_seg"
import random


stopword_path = '/Users/liyangyang/Downloads/stopwords/stopwords1893.txt'
import jieba


def stopwordslist():
    # stopwords = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]
    stopwords = ['，', '。', '、', '...', '“', '”', '《', '》', '：', '；']
    return stopwords

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # text = pd.read_csv(path + data_file)
    # x_text = np.array(text[column])
    #
    # # y=[]
    # # for i in range(len(text['class'])):
    # # print(text['class'])
    # # y = dense_to_one_hot(text['class'],19)
    # y = (text["class"]-1).astype(int)
    # y = np.array(text["id"])

    train = pd.read_csv(path + 'train.csv')
    random.shuffle(train)
    train_doc_list = []
    for i in range(len(train)):
        sentence_seged = jieba.cut(train['content'][i].strip())
        # sentence_seged = split_word(train['content'][i].strip())
        stopwords = stopwordslist()
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if (word != '\t') & (word.strip() != ''):
                    outstr += word
                    outstr += '\t'
        if (outstr == ''):
            outstr = 'NaN'
        train_doc_list.append(outstr)
    x_train = np.array(train_doc_list)

    train.loc[train['subject'] == '动力', 'subject'] = 0
    train.loc[train['subject'] == '价格', 'subject'] = 1
    train.loc[train['subject'] == '内饰', 'subject'] = 2
    train.loc[train['subject'] == '配置', 'subject'] = 3
    train.loc[train['subject'] == '安全性', 'subject'] = 4
    train.loc[train['subject'] == '外观', 'subject'] = 5
    train.loc[train['subject'] == '操控', 'subject'] = 6
    train.loc[train['subject'] == '油耗', 'subject'] = 7
    train.loc[train['subject'] == '空间', 'subject'] = 8
    train.loc[train['subject'] == '舒适性', 'subject'] = 9
    y_train = train['subject']
    return x_train, y_train

def load_dev_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    text = pd.read_csv(path + 'train_set.csv')
    x_text = np.array(text[column])
    # y=[]
    # for i in range(len(text['class'])):
    # print(text['class'])
    y = np.array(text['class'])
    return x_text, y


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

from tensorflow.contrib import learn

def test():
    x_text, y = load_data_and_labels('train_set.csv')
    vocab_processor = learn.preprocessing.VocabularyProcessor(2000,min_frequency=3)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    vocab_size = len(vocab_processor.vocabulary_)
    print(vocab_size)

# test()


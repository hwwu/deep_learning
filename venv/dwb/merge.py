#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/5 上午11:17
# @Author   :hwwu
# @File     :merge.py

import pandas as pd, numpy as np

path = '/Users/liyangyang/Downloads/dwb/new_data/'

def write_result(id, predictions):
    r_id = []
    r_predictions = []
    for i in range(len(id)):
        r_id.append(int(id[i]))
        r_predictions.append(int(predictions[i]))

    english_column = pd.Series(r_id, name='id')
    number_column = pd.Series(r_predictions, name='class')
    predictions = pd.concat([english_column, number_column], axis=1)
    predictions.to_csv(path + 'merge_result_data.csv', index=0, sep=',', columns=['id', 'class'])


r75 = pd.read_csv(path+'result_data.csv')['class']
rcnn = pd.read_csv(path+'p4_cnn_result_data.csv')['class']
rrnn = pd.read_csv(path+'result_rnn.csv')['class']

id = []
predictions =[]
for i in range(len(r75)):
    id.append(i)
    if (rcnn[i]==rrnn[i]):
        predictions.append(rcnn[i])
    else:
        predictions.append(r75[i])

write_result(id,predictions)







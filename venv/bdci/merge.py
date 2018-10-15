#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/12 下午3:48
# @Author   :hwwu
# @File     :merge.py

import pandas as pd, numpy as np

path = '/Users/liyangyang/Downloads/bdci/'

def write_result(id, predictions):
    r_id = []
    r_predictions = []
    for i in range(len(id)):
        r_id.append(str(id[i]))
        r_predictions.append(int(predictions[i]))

    english_column = pd.Series(r_id, name='content_id')
    number_column = pd.Series(r_predictions, name='sentiment_value')
    predictions = pd.concat([english_column, number_column], axis=1)
    predictions.to_csv(path + 'merge_result_data_sentiment_value.csv', index=0, sep=',', columns=['content_id', 'sentiment_value'])


# r75 = pd.read_csv(path+'MultinomialNB.csv')['sentiment_value']
# rcnn = pd.read_csv(path+'LinearSVC.csv')['sentiment_value']
# rrnn = pd.read_csv(path+'RandomForestClassifier.csv')['sentiment_value']
# print('r75.shape',r75.shape)
# print('rcnn.shape',rcnn.shape)
# print('rrnn.shape',rrnn.shape)
#
# id = pd.read_csv(path+'MultinomialNB.csv')['content_id']
# predictions =[]
# for i in range(len(r75)):
#     # id.append(r75['content_id'][i])
#     if (rcnn[i]==rrnn[i]):
#         predictions.append(rcnn[i])
#     else:
#         predictions.append(r75[i])
#
# write_result(id,predictions)

sentiment_value = pd.read_csv(path+'merge_result_data_sentiment_value.csv')
subject = pd.read_csv(path+'merge_result_data_subject.csv')
content = pd.read_csv(path+'train.csv')

subject.loc[subject['subject'] == 0, 'subject'] = '动力'
subject.loc[subject['subject'] == 1, 'subject'] = '价格'
subject.loc[subject['subject'] == 2, 'subject'] = '内饰'
subject.loc[subject['subject'] == 3, 'subject'] = '配置'
subject.loc[subject['subject'] == 4, 'subject'] = '安全性'
subject.loc[subject['subject'] == 5, 'subject'] = '外观'
subject.loc[subject['subject'] == 6, 'subject'] = '操控'
subject.loc[subject['subject'] == 7, 'subject'] = '油耗'
subject.loc[subject['subject'] == 8, 'subject'] = '空间'
subject.loc[subject['subject'] == 9, 'subject'] = '舒适性'

# df = pd.DataFrame({"content_id": sentiment_value['content_id'], "subject": subject['subject'],'sentiment_value':sentiment_value['sentiment_value'].astype(int),'sentiment_word':''})
# df.to_csv(path+'result.csv', index = False, header=True,encoding='UTF-8')

content_id = sentiment_value['content_id']
subject = subject['subject']
sentiment_value = sentiment_value['sentiment_value'].astype(int)
sentiment_word = content['sentiment_word'][:len(content_id)]
print('sentiment_value',sentiment_value.shape)
predictions = pd.concat([content_id, subject,sentiment_value,sentiment_word], axis=1)
predictions.to_csv(path + 'result.csv', index=0, sep=',', columns=['content_id', 'subject','sentiment_value','sentiment_word'],encoding='UTF-8')
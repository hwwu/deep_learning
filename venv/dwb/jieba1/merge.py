#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/10 下午6:49
# @Author   :hwwu
# @File     :merge.py

import pandas as pd
import numpy as np

path = '/Users/liyangyang/Downloads/jieba/data/'

result = pd.read_csv(path + 'result.csv')
kw = pd.read_csv(path + 'train_docs_keywords.txt', sep='\t', header=None)
kw.columns = ['id', 'lw']
# print(kw)
result['label1']=result['label1'].replace('','nan')
result['label2']=result['label2'].replace('','nan')
id = []
label1 = []
label2 = []
print(len(result))
for i in range(len(result)):
    m_id = str(result['id'][i])
    if (m_id=='D101107'):
        print(m_id,str(result['label1'][i]))
        print(m_id,str(result['label2'][i]))
    id.append(m_id)
    l1 = ''
    l2 = ''
    for j in range(len(kw)):
        if (m_id == kw['id'][j]):
            words = str(kw['lw'][j]).split(',')
            if(len(words)>1):
                l1 = words[0]
                l2 = words[1]
            else:
                l1 = words[0]
                l2 = 'nan'
            print(m_id,l1,l2)
    if (l1 != ''):
        label1.append(l1.replace(',',''))
        label2.append(l2.replace(',',''))
    else:
        label1.append(str(result['label1'][i]).replace(',',''))
        label2.append(str(result['label2'][i]).replace(',',''))

id_column = pd.Series(id, name='id')
label1_column = pd.Series(label1, name='label1')
label2_column = pd.Series(label2, name='label2')
predictions = pd.concat([id_column, label1_column, label2_column], axis=1)
predictions.to_csv(path + 'merge_result_data.csv', index=0, sep=',', columns=['id', 'label1', 'label2'],encoding='UTF-8')


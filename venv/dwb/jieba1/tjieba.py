#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/10 下午1:42
# @Author   :hwwu
# @File     :jieba1.py

path = '/Users/liyangyang/Downloads/jieba/data/'
file = 'all_docs.txt'

import numpy as np
import pandas as pd

data = pd.read_csv(path+file,sep='\001',header=None)
data.columns = ['id','title','doc']
# # print(data['doc'])
# new_data = data['title']+data['doc']
# print(new_data)
#
# regex = analyse.extract_tags(new_data,topK=4,withWeight=False,allowPOS=())
#
# print(regex)

print(len(data))

import codecs
import os

import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

base_path = path + file
seg_path = path + 'all_docs_seg.txt'

stopword_path='/Users/liyangyang/Downloads/stopwords/CNENstopwords.txt'

def stopwordslist():
    stopwords = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]
    return stopwords


def segment():
    """word segment"""
    fw = codecs.open(seg_path, 'w', 'utf-8')
    doc_list=[]
    for i in range(len(data)):
        title = str(data['title'][i])
        doc = str(data['doc'][i])
        # row = line.split('\001')
        # seg_list = jieba.cut(line.strip())
        sentence_seged = jieba.cut((title+ '。' + doc).strip())
        stopwords = stopwordslist()
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr +='\t'
        l = outstr
        doc_list.append(l)
        fw.write(l)
        fw.write('\n')
    fw.flush()
    fw.close()
    return doc_list

def tfidf_top(trade_list, doc_list, max_df, topn):
    vectorizer = TfidfVectorizer(max_df=max_df,min_df=1,use_idf=1,smooth_idf=1, sublinear_tf=1)
    matrix = vectorizer.fit_transform(doc_list)
    feature_dict = {v: k for k, v in vectorizer.vocabulary_.items()}  # index -> feature_name
    top_n_matrix = np.argsort(-matrix.todense())[:, :topn]  # top tf-idf words for each row
    df = pd.DataFrame(np.vectorize(feature_dict.get)(top_n_matrix), index=trade_list)  # convert matrix to df
    return df


# dl = segment()
# print('first')
# tl = data['id']
# tdf = tfidf_top(tl, dl, max_df=0.5, topn=2)
# print('second')
# tdf.to_csv(path+'resilt.csv', header=False, encoding='utf-8')
# print('done')



# fw = codecs.open(path+'result.csv', 'w', 'utf-8')
#
# fw.write("id,label1,label2"+"\n")
#
#
# def textrank():
#     n = 0
#     fw = codecs.open(seg_path, 'w', 'utf-8')
#     # doc_list = []
#     # for i in range(len(data)):
#     for i in range(100):
#         # n+=1
#         # title = str(data['title'][i])
#         # doc = str(data['doc'][i])
#         row = line.split('\001')
#         seg_list = jieba.cut(line.strip())
#         # sentence_seged = jieba.cut((title + '。' + doc).strip())
#         # stopwords = stopwordslist()
#         # outstr = ''
#         # for word in sentence_seged:
#         #     if word not in stopwords:
#                 if word != '\t':
#                     outstr += word
#                     outstr += '\t'
#         keywords = analyse.textrank(outstr, topK=2, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
#         if (len(keywords)==0):
#             fw.write(str(data['id'][i]) + "," + str('') + "," + str('') + "\n")
#         elif (len(keywords)==1):
#             fw.write(str(data['id'][i]) + "," + str(keywords[0]) + "," + str('') + "\n")
#         else:
#             fw.write(str(data['id'][i]) + "," + str(keywords[0]) + "," + str(keywords[1]) + "\n")
#
#         if(n%1000==0):
#             print('flush',n/1000)
#             fw.flush()
# textrank()

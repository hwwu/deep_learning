#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/12 下午1:48
# @Author   :hwwu
# @File     :read_data.py

path = '/Users/liyangyang/Downloads/bdci/'

import pandas as pd, numpy as np

train = pd.read_csv(path + 'train.csv')[:2000]
test = pd.read_csv(path + 'test_public.csv')[:1000]

# y_train = train['sentiment_value'].astype(int)
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


print(train.shape)
print(test.shape)

stopword_path = '/Users/liyangyang/Downloads/stopwords/stopwords1893.txt'
import jieba


def stopwordslist():
    stopwords = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]
    # stopwords = ['，', '。', '、', '...', '“', '”', '《', '》', '：', '；']
    return stopwords


def split_word(line):
    result = []
    for i in range(len(line)):
        result.append(line[i:i + 1])
    return result

import codecs
f = codecs.open(path+'train_no_lable.txt', 'a', 'utf8')
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
                # outstr += '\t'
                outstr += ' '
    # if (outstr == ''):
    #     outstr = 'NaN'
    # outstr +='__myprefix__'
    # outstr +=str(y_train[i])
    f.write(outstr+'\n')
    train_doc_list.append(outstr)

train_doc_list = np.array(train_doc_list)
print(train_doc_list.shape)

test_doc_list = []
for i in range(len(test)):
    sentence_seged = jieba.cut(test['content'][i].strip())
    # sentence_seged = split_word(test['content'][i].strip())
    stopwords = stopwordslist()
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += '\t'
    if (outstr == ''):
        outstr = 'NaN'
    test_doc_list.append(outstr)
test_doc_list = np.array(test_doc_list)
print(test_doc_list.shape)
#
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#
# count_vec = CountVectorizer(analyzer='word')
# data_train_count = count_vec.fit_transform(train_doc_list)
# data_test_count = count_vec.transform(test_doc_list).toarray()
# #词汇表
# print('\nvocabulary list:\n\n',count_vec.get_feature_names())
# print( '\nvocabulary dic :\n\n',count_vec.vocabulary_)
# print ('vocabulary:\n\n')
# for key,value in count_vec.vocabulary_.items():
#     print(key,value)
# print('.............')
# print(data_train_count)

tfidf = TfidfVectorizer(
    ngram_range=(1, 1),  # 二元文法模型
    use_idf=1,
    # analyzer='char',
    smooth_idf=1)

data_train_count_tf = tfidf.fit_transform(train_doc_list)
data_test_count_tf = tfidf.transform(test_doc_list)

print('\nvocabulary list:\n\n',tfidf.get_feature_names())
print( '\nvocabulary dic :\n\n',tfidf.vocabulary_)
print ('vocabulary:\n\n')
for key,value in tfidf.vocabulary_.items():
    print(key,value)
print('.............')
print(type(data_train_count_tf))
#
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import cross_val_score
#
# clf = MultinomialNB()
# clf.fit(data_train_count, y_train)
# print("多项式贝叶斯分类器20折交叉验证得分: ", np.mean(cross_val_score(clf, data_train_count, y_train, cv=10, scoring='accuracy')))
# clf.fit(data_train_count_tf, y_train)
# print("多项式贝叶斯分类器TFIDF,20折交叉验证得分: ",
#       np.mean(cross_val_score(clf, data_train_count_tf, y_train, cv=10, scoring='accuracy')))
# # clf_pred = clf.predict(data_test_count_tf)
# # df = pd.DataFrame({"content_id": test['content_id'], "sentiment_value": clf_pred})
# # df.to_csv(path+'MultinomialNB.csv', index = False, header=True)
# #
# from sklearn import svm
#
# lin_clf = svm.LinearSVC(class_weight='balanced')
# lin_clf.fit(data_train_count, y_train)
# print("svm分类器20折交叉验证得分: ", np.mean(cross_val_score(lin_clf, data_train_count, y_train, cv=10, scoring='accuracy')))
# lin_clf.fit(data_train_count_tf, y_train)
# print("svm分类器TFIDF,20折交叉验证得分: ",
#       np.mean(cross_val_score(lin_clf, data_train_count_tf, y_train, cv=10, scoring='accuracy')))
# # lin_clf_pred = lin_clf.predict(data_test_count_tf)
# # df = pd.DataFrame({"content_id": test['content_id'], "sentiment_value": lin_clf_pred})
# # df.to_csv(path+'LinearSVC.csv', index = False, header=True)
#
# from sklearn.ensemble import RandomForestClassifier
#
# lin_forest = RandomForestClassifier(n_estimators=10, random_state=1, class_weight='balanced')
# lin_forest.fit(data_train_count, y_train)
# print("RandomForestClassifier分类器20折交叉验证得分: ",
#       np.mean(cross_val_score(lin_forest, data_train_count, y_train, cv=10, scoring='accuracy')))
# lin_forest.fit(data_train_count_tf, y_train)
# print("RandomForestClassifier分类器TFIDF,20折交叉验证得分: ",
#       np.mean(cross_val_score(lin_forest, data_train_count_tf, y_train, cv=10, scoring='accuracy')))
# # lin_forest_pred = lin_forest.predict(data_test_count_tf)
# # df = pd.DataFrame({"content_id": test['content_id'], "sentiment_value": lin_forest_pred})
# # df.to_csv(path+'RandomForestClassifier.csv', index = False, header=True)
#
#
# import xgboost as xgb
#
# model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468)
# model_xgb.fit(data_train_count, y_train)
# print("model_xgb分类器20折交叉验证得分: ",
#       np.mean(cross_val_score(model_xgb, data_train_count, y_train, cv=10, scoring='accuracy')))
# model_xgb.fit(data_train_count_tf, y_train)
# print("model_xgb分类器TFIDF,20折交叉验证得分: ",
#       np.mean(cross_val_score(model_xgb, data_train_count_tf, y_train, cv=10, scoring='accuracy')))
# # model_xgb_pred = model_xgb.predict(data_test_count_tf)
# # df = pd.DataFrame({"content_id": test['content_id'], "sentiment_value": model_xgb_pred})
# # df.to_csv(path+'XGBClassifier.csv', index = False, header=True)

#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/8/20 上午11:28
# @Author   :hwwu
# @File     :baseline.py
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from tensorflow.contrib import learn

path = '/Users/liyangyang/Downloads/dwb/new_data/'
# column = "word_seg"
# train = pd.read_csv(path+'train_set.csv')
test = pd.read_csv(path+'test_set.csv')
test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(3,4),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
#
# train = np.array(train[column])
# test = np.array(test[column])
#
# vocab_processor_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/testcnn/vocab-5000/'
# vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_processor_path)
# train = np.array(list(vocab_processor.transform(train)))
# test = np.array(list(vocab_processor.transform(test)))
# #
# # train = pd.DataFrame(train)
# # test = pd.DataFrame(test)
#
# np.save(path+'vocab/vocab_train',train)
# np.save(path+'vocab/vocab_test',test)


# t1 = np.load(path+'vocab/vocab_train.npy')
# t2 = np.load(path+'vocab/vocab_test.npy')
#
# train=[]
# test=[]
# for i in range(len(t1)):
#     row = str(t1[i][0])
#     for j in range(1,len(t1[i])):
#         s = str(t1[i][j])
#         if (s!='0'):
#             row = row + '\t' + s
#     train.append(row)
# print(train[0])
#
# for i in range(len(t2)):
#     row = str(t2[i][0])
#     for j in range(1,len(t2[i])):
#         s = str(t2[i][j])
#         if (s != '0'):
#             row = row + '\t' + s
#     test.append(row)
# print(test[0])
#
# train = np.array(train)
# test = np.array(test)
# print(train.shape)
# print(test.shape)
#
# np.save(path+'vocab/vocab_train_1',train)
# np.save(path+'vocab/vocab_test_1',test)

t1 = np.load(path+'vocab/vocab_train_1.npy')
train = np.array(t1)

print('start tf-idf fit')
trn_term_doc = vec.fit_transform(train)
np.savetxt(path+'tf-idf/train_data',trn_term_doc)

print('start tf-idf transform')
t2 = np.load(path+'vocab/vocab_test_1.npy')
test = np.array(t2)
test_term_doc = vec.transform(test)
print('tf-idf transform done')

fid0=open(path+'baseline_time.csv','w')
np.savetxt(path+'tf-idf/test_data',test_term_doc)
print('save data done')

y=(train["class"]-1).astype(int)
print('start fit')
lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc[:80000],y[:80000])
# preds = lin_clf.predict(test_term_doc)
# lin_forest = RandomForestClassifier(n_estimators=100, random_state=1)
# lin_forest.fit(trn_term_doc,y)
print('fit done')
print('start predict')
pred = lin_clf.score(trn_term_doc[80000:],y[80000:])
print('predict done')
print(pred)

preds = lin_clf.predict(test_term_doc)
i=0
fid0.write("id,class"+"\n")
for item in preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()


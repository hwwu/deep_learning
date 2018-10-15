#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/8/15 下午4:30
# @Author   :hwwu
# @File     :find_base_feature.py

import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

dic = [22, 135, 591, 592, 593, 594, 595, 737, 948, 1070, 1173, 1175, 1286,
       1362, 1451, 1519, 1565, 1666, 1717, 1894, 2137, 2223, 2271, 2414,
       2579, 2797, 2875, 2916, 2986, 2684, 3723, 3597, 3599, 3603, 3605,
       3607, 3610, 3601, 3602, 3421, 3393, 3538, 3539, 3540, 5521, 6016,
       7437, 11832, 15355, 3152, 3612, 3611]

path = '/Users/liyangyang/Downloads/datafountain/guangdianfute/'
file = 'public.train.csv'
data = pd.read_csv(path + file)
data = data[(data['平均功率'] < 10000.0)]
data = data[(data['现场温度'] > -1000.0)]
data = data[~((data['板温'] == 0.01) & (data['现场温度'] == 0.1))]
data = data[~(data['ID'].isin(dic))]

print(data.max())

feature_name = [i for i in data.columns if i!='发电量']
feature_name = [i for i in feature_name if i!='ID']
train_data = data[feature_name]
train_label = data['发电量']

# from sklearn import preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()
# train_data = min_max_scaler.fit_transform(train_data)
# print(feature_name)

#方差选择法
# from sklearn.feature_selection import VarianceThreshold
# print(VarianceThreshold(threshold=0.03).fit_transform(train_data)[0])

#相关系数法
# from sklearn.feature_selection import SelectKBest
# from scipy.stats import pearsonr
# print(feature_name)
# print(train_data[0])
# from sklearn.feature_selection import f_regression,mutual_info_regression
# for i in range(1,20):
#        print(SelectKBest(f_regression, k=i).fit_transform(train_data, train_label)[0])
#        print(SelectKBest(mutual_info_regression, k=i).fit_transform(train_data, train_label)[0])

#Pearson相关系数
# from scipy.stats import pearsonr
# for i in range(0,19):
#        print(i,pearsonr(train_data[:,i], train_label))

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
print(train_data.head())
train_data = np.array(train_data)
train_label = np.array(train_label)
print(RFE(estimator=LinearRegression(), n_features_to_select=10).fit_transform(train_data, train_label)[0])


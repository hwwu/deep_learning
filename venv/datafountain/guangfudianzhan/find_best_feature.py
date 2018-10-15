#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/8/14 上午10:17
# @Author   :hwwu
# @File     :find_best_feature.py

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
data = data[(data['转换效率'] < 500.0)]
data = data[~((data['板温'] == 0.01) & (data['现场温度'] == 0.1))]
data = data[~(data['ID'].isin(dic))]

train = data[::1]
test = data[::1]

feature_name = [i for i in data.columns if i != '发电量']
feature_name = [i for i in feature_name if i != 'ID']
feature_name = [i for i in feature_name if i != '现场温度']
# feature_name = [i for i in feature_name if i != '转换效率']
# feature_name = [i for i in feature_name if i != '功率A']
# feature_name = [i for i in feature_name if i != '功率B']
# feature_name = [i for i in feature_name if i != '功率C']

train_data = train[feature_name]
train_label = train['发电量']
test_data = test[feature_name]
train_label = np.array(train_label)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train_data = min_max_scaler.fit_transform(train_data)
train_data = DF(train_data, columns=(i for i in feature_name))
print(train_data.head())

xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.005, max_depth=23,
                             max_delta_step=100000,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)


def rmse_my(y_test, y_):
    error = []
    for i in range(len(y_test)):
        error.append(y_test[i] - y_[i])

    squaredError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
    from math import sqrt
    RMSE = sqrt(sum(squaredError) / len(squaredError))
    print("RMSE = ", RMSE)  # 均方根误差RMSE
    return RMSE


def get_division_feature(data, feature_name):
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns) - 1):
        for j in range(i + 1, len(data[feature_name].columns)):
            new_feature_name.append(data[feature_name].columns[i] + '/' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '*' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '+' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '-' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i]] / data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i]] * data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i]] + data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i]] - data[data[feature_name].columns[j]])

    temp_data = DF(pd.concat(new_feature, axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([temp_data], axis=1).reset_index(drop=True)
    # print(data.shape)
    return data.reset_index(drop=True)


def get_square_feature(data, feature_name):
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns)):
        new_feature_name.append(data[feature_name].columns[i] + '**2')
        new_feature_name.append(data[feature_name].columns[i] + '**1/2')
        new_feature.append(data[data[feature_name].columns[i]] ** 2)
        new_feature.append(data[data[feature_name].columns[i]] ** (1 / 2))
    temp_data = DF(pd.concat(new_feature, axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([temp_data], axis=1).reset_index(drop=True)
    # print(data.shape)
    return data.reset_index(drop=True)


def find_best_feature(feature_name):
    get_ans_face = feature_name
    xgb_model.fit(train_data[get_ans_face], train_label)
    y_ = xgb_model.predict(train_data[get_ans_face])
    m = rmse_my(train_label, y_)
    return m


train_datatrain_d = get_square_feature(train_data, feature_name)
train_data_division = get_division_feature(train_data, feature_name)
train_data = pd.concat([train_datatrain_d, train_data_division, train_data], axis=1)
feature_name = [i for i in train_data.columns]
print(train_data.shape)

print(feature_name)

now_feature = []
# check = 0.05416978387299058
# d = [1,2,3,5,6,7,8,9,10,21,22,27,31,32,33,34,35,36,37,38,39,40,42,43,44,46,47,48,49,55,56,60,61,65,66,78,79,80,82,
#      103,104,108,109,110,111,112,128,129,130,131,214,215,221,222,243,247,248,251,252]
#
# for i in d:
#     now_feature.append(feature_name[i-1])
# for i in range(354,len(feature_name)):
# check = 0.05878801229207516
d = [1, 2, 3, 4, 5, 6, 7, 9, 10, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 82, 83, 84, 85, 86, 87,
     89, 90, 94, 95, 96, 97, 98, 122, 123, 124, 125, 126, 127, 128, 129, 132, 133, 136, 137, 163, 164, 167, 168, 170,
     171, 172, 176, 177, 179, 180, 214]

# for i in d:
#     now_feature.append(feature_name[i - 1])
# for i in range(324, len(feature_name)):
#     now_feature.append(feature_name[i])
#     jj = find_best_feature(now_feature)
#     if jj < check:
#         print('目前特征长度为', len(now_feature), ' 目前帅气的RSME为值是', jj, ' 成功加入第', i + 1, '个', 'RSME降低', check - jj)
#         check = jj
#     else:
#         print('尝试加入第', i + 1, '个特征失败')
#         now_feature.pop()
#     print(now_feature)
#
now_feature2 = []
check = 100
for i in range(len(feature_name)):
    now_feature2.append(feature_name[len(feature_name)-i-1])
    jj = find_best_feature(now_feature2)
    if jj<check:
        print('目前特征长度为',len(now_feature2),' 目前帅气的cv值是',jj,' 成功加入第',i+1,'个','增值为',check-jj)
        check = jj
    else:
        print('尝试加入第', i + 1, '个特征失败')
        now_feature2.pop()
    print(now_feature2)

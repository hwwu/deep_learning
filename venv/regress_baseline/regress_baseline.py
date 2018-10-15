#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/10/9 下午4:14
# @Author   :hwwu
# @File     :regress_baseline.py
import pandas as pd
import numpy as np
import gc
import os
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
import xgboost as xgb
import math
from sklearn.utils import shuffle

warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()

print('train_x,train_y should be pandas DataFrame')

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import sys

path = '/Users/liyangyang/PycharmProjects/mypy/deep_learning/venv/datafountain/guangfudianzhan/'
sys.path.append(path)
import read_data

dis = [1, 190, 379, 567, 755, 940, 1123, 1314, 1503, 1505, 1694, 1879,
       2070, 2257, 2444, 2632, 2823, 3013, 3202, 3379, 3567, 3746, 3927, 4089,
       4278, 4459, 4648, 4652, 4821, 5010, 5013, 5017, 5059, 5061, 5069, 5074,
       5077, 5281, 5285, 5287, 5292, 5508, 5703, 5911, 5913, 5916, 5918, 6121,
       6337, 6524, 6528, 6531, 6534, 6723, 6923, 7116, 7326, 7535, 7740, 7937,
       8146, 8245, 8258, 8310, 8488, 8705, 8711, 8878, 9088, 9296, 9505, 9719,
       9916, 10124, 10335, 10544, 10736, 10914, 10917, 11119, 11331, 11540,
       11753, 11963, 12170, 12381, 12592, 12802, 13009, 13214, 13426, 13617,
       13830, 14032, 14243, 14457, 14666, 14882, 15091, 15299, 15508, 15719,
       15937, 16144, 16348, 16540, 16747, 16925, 17133, 17342,
       17527, 17543, 17745, 17876]

dic = [22, 135, 591, 592, 593, 594, 595, 737, 948, 1070, 1173, 1175, 1286,
       1362, 1451, 1519, 1565, 1666, 1717, 1894, 2137, 2223, 2271, 2414,
       2579, 2797, 2875, 2916, 2986, 2684, 3723, 3597, 3599, 3603, 3605,
       3607, 3610, 3601, 3602, 3421, 3393, 3538, 3539, 3540, 5521, 6016,
       7437, 11832, 15355, 3152, 3612, 3611]
from sklearn import preprocessing


def load_train_data():
    min_max_scaler = preprocessing.MinMaxScaler()
    train_ = read_data.read_result_data('public.train.csv')
    train_x = train_[:, 2:21]
    train_y = train_[:, 21]
    train_x = min_max_scaler.fit_transform(train_x)

    train_z = train_[:, 1]

    train_len = len(train_y)
    train_y.shape = (1, train_len)
    train_y = np.transpose(train_y)

    x, y = [], []
    for i in range(train_len):
        if ((round(train_x[i][0], 2) != 0.01) | (round(train_x[i][1], 1) != 0.1)):

            id = 0.0
            for j in range(len(dis)):
                if (train_z[i] < dis[j]):
                    id = 0.5 - np.abs((int(train_z[i]) - dis[j - 1]) / (dis[j] - dis[j - 1]) - 0.5)
                    break

            if (train_z[i] not in dic):
                # x.append(character(id,train_x[i]))
                x.append([id,
                          train_x[i][0],
                          train_x[i][1],
                          train_x[i][3],
                          train_x[i][2] * train_x[i][4],
                          train_x[i][2] * train_x[i][5],
                          train_x[i][2] * train_x[i][6],
                          train_x[i][13],
                          train_x[i][14],
                          train_x[i][15],
                          train_x[i][16],
                          train_x[i][17],
                          train_x[i][18]
                          ])
                y.append(abs(train_y[i]))
    print(len(x))
    return x, y


x, y = load_train_data()

X_train = x[0:8000:1]
y_train = y[0:8000:1]
X_test = x[8000:8905:1]
y_test = y[8000:8905:1]
train_x = pd.DataFrame(X_train)
train_y = pd.DataFrame(y_train)
test = pd.DataFrame(X_test)

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1, max_iter=100000))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3, max_iter=100000))
KRR = KernelRidge(alpha=0.6, kernel='linear', degree=2, coef0=2.5)
xg = xgb.XGBRegressor(objective='reg:linear', booster='gbtree', colsample_bytree=0.4603, gamma=0.0468,
                      learning_rate=0.01, max_depth=23,
                      max_delta_step=100000,
                      min_child_weight=57, n_estimators=2200,
                      reg_alpha=0.4640, reg_lambda=0.8571,
                      subsample=0.5213,
                      random_state=777, n_jobs=-1)
lg = lgb.LGBMRegressor(
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=1000,
    subsample=.9,
    colsample_bytree=.9,
    random_state=666
)

global_num = 0


def get_column_index():
    global global_num
    global_num += 1
    return str(global_num)


# 切分训练集
def split_data(train_x, train_y, n_splits):
    train_y.columns = ['hw_id']
    train = pd.concat([train_y, train_x], axis=1)
    train = shuffle(train)
    train_y = train['hw_id']
    train_x = train.drop(['hw_id'], axis=1)
    del train
    size = math.ceil(len(train_x) / n_splits)
    fold_ids = []
    for i in range(n_splits):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < len(train_x) else len(train_x)
        val_x_data = train_x[start:end]
        val_y_data = train_y[start:end].tolist()
        train_x_data = pd.concat([train_x[0:start], train_x[end:len(train_x)]])
        train_y_data = pd.concat([train_y[0:start], train_y[end:len(train_x)]]).tolist()
        fold_ids.append([train_x_data, val_x_data, train_y_data, val_y_data])
    return fold_ids


# def LGB_predict(x_train, train_x, val_x, train_y, val_y, test, index):
#     lgb.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric='rmse', verbose=1, early_stopping_rounds=300)
#     val = lgb.predict(x_train)
#     res = lgb.predict(test)
#     print('lgb ' + str(index) + ' predict finish!')
#     rmse_my(y_test, res, lgb, 10 + index)
#     gc.collect()
#     return pd.DataFrame(val), pd.DataFrame(res)
#
#
# def XGB_predict(x_train, train_x, val_x, train_y, val_y, test, index):
#     xgb.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric='rmse', verbose=1, early_stopping_rounds=300)
#     val = xgb.predict(x_train)
#     res = xgb.predict(test)
#     print('xgb ' + str(index) + ' predict finish!')
#     rmse_my(y_test, res, xgb, 10 + index)
#     gc.collect()
#     return pd.DataFrame(val), pd.DataFrame(res)


def model_predict(model, x_train, train_x, val_x, train_y, val_y, test, index):
    if model in [lg, xg]:
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric='rmse', verbose=1, early_stopping_rounds=100)
    else:
        model.fit(train_x, train_y)
    val = model.predict(x_train)
    res = model.predict(test)
    print(str(model) + str(index) + ' predict finish!')
    rmse_my(y_test, res, model, index)
    gc.collect()
    return pd.DataFrame(val), pd.DataFrame(res)


def f1_predict(models, x_train, y_train, test, n_splits=5):
    train = pd.DataFrame()
    result = pd.DataFrame()
    fold_ids = split_data(x_train, y_train, n_splits)
    for k, model in enumerate(models):
        for i, fold in enumerate(fold_ids):
            train_x, val_x, train_y, val_y = fold
            val, res = model_predict(model, x_train, train_x, val_x, train_y, val_y, test, i)
            col = get_column_index()
            val.columns = ['pred_' + col]
            res.columns = ['pred_' + col]
            train = pd.concat([train, val], axis=1)
            result = pd.concat([result, res], axis=1)
    return train, result


def middle_predict(models, train_x, train_y, test, n_splits=5):
    #####   first  Regressor
    val, res = f1_predict(models, train_x, train_y, test, n_splits)
    train_x = pd.concat([train_x, val], axis=1)
    test = pd.concat([test, res], axis=1)
    col = res.columns
    c1 = get_column_index()
    train_x['mean_' + c1] = val[col].mean(axis=1)
    train_x['median' + c1] = val[col].median(axis=1)
    test['mean_' + c1] = res[col].mean(axis=1)
    test['median' + c1] = res[col].median(axis=1)
    return train_x, train_y, test


def result_predict(models, train_x, train_y, test, n_splits=5):
    #####   second  Regressor
    val, res = f1_predict(models, train_x, train_y, test, n_splits)
    col = res.columns
    result = res[col].mean(axis=1)
    return result


my_best_model = xg
my_best_score = 999.

score_list = []


def rmse_my(y_test, y_, model, index):
    global my_best_model
    global my_best_score
    error = []
    n = 0
    for i in range(len(y_test)):
        if ((y_test[i] - y_[i]) * (y_test[i] - y_[i]) > 1):
            # print(y_test[i], y_[i])
            n += 1
        error.append(y_test[i] - y_[i])
    # print('n', n)

    squaredError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方

    from math import sqrt
    score = sqrt(sum(squaredError) / len(squaredError))
    if score < my_best_score:
        my_best_score = score
        my_best_model = model
        pd.DataFrame(y_).to_csv(os.getcwd() + '/result_' + str(score) + '.csv', index=False, sep=',', encoding='utf8')
    score_list.append([str(model)[:str(model).index("(")], index, score])
    print(str(model)[:str(model).index("(")], index, "RMSE = ", score)  # 均方根误差RMSE


train_x, train_y, test = middle_predict([xg, lasso], train_x, train_y, test, n_splits=5)

r1 = result_predict([lg], train_x, train_y, test, n_splits=3)

train_x, train_y, test = middle_predict([lg, ENet, KRR], train_x, train_y, test, n_splits=5)

r2 = result_predict([xg], train_x, train_y, test, n_splits=3)

rmse_my(y_test, r1, 'r1(', 0)
rmse_my(y_test, r2, 'r2(', 0)
rmse_my(y_test, r1 * 0.5 + r2 * 0.5, 'merge(', 0)

print(my_best_model)
print(my_best_score)

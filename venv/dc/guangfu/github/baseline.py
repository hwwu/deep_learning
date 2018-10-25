#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/10/25 上午11:07
# @Author   :hwwu
# @File     :baseline.py

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures

path = './dc/guangfu/'


def get_hour(x):
    h = int(x[11:13])
    m = int(x[14:16])
    if m in [14, 29, 44]:
        m += 1
    if m == 59:
        m = 0
        h += 1
    if h == 24:
        h = 0
    return h * 60 + m


def add_poly_features(data, column_names):
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1, col, poly_features[col])
    return rest_features


train_x_old = pd.read_csv(path + 'train_1.csv')
test = pd.read_csv(path + 'test_1.csv')
train_x_old['month'] = train_x_old['时间'].apply(lambda x: x[5:7]).astype('int32')
train_x_old['day'] = train_x_old['时间'].apply(lambda x: x[8:10]).astype('int32')
train_x_old['hour'] = train_x_old['时间'].apply(lambda x: get_hour(x)).astype('int32')
test['month'] = test['时间'].apply(lambda x: x[5:7]).astype('int32')
test['day'] = test['时间'].apply(lambda x: x[8:10]).astype('int32')
test['hour'] = test['时间'].apply(lambda x: get_hour(x)).astype('int32')

train_y = train_x_old['实际功率']
train_x = train_x_old.drop(['实发辐照度', '实际功率'], axis=1)
train_x['dis2peak'] = train_x['hour'].apply(lambda x: (810 - abs(810 - x)) / 810)
train_x = add_poly_features(train_x, ['风速', '风向'])
train_x = add_poly_features(train_x, ['温度', '压强', '湿度'])

id = test['id']
del_id = test[test['辐照度'].isin([-1.0])]['id']
test = test.drop(['id'], axis=1)
test['dis2peak'] = test['hour'].apply(lambda x: (810 - abs(810 - x)) / 810)
test = add_poly_features(test, ['风速', '风向'])
test = add_poly_features(test, ['温度', '压强', '湿度'])

train_x = train_x.drop(['时间'], axis=1)
test = test.drop(['时间'], axis=1)
print('train_x.shape,test_1.shape : ', train_x.shape, test.shape)

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=678)

params = {
    "objective": "regression",
    "metric": "mse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 666,
    "verbosity": -1
}


def lgb_train():
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    print('begin train')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=100)
    # y_pred = gbm.predict(X_test)
    ##write result
    republish_pred = gbm.predict(test)
    republish_pred = pd.DataFrame(republish_pred)
    sub = pd.concat([id, republish_pred], axis=1)
    print(sub.shape)
    sub.columns = ['id', 'predicition']
    sub.loc[sub['id'].isin(del_id), 'predicition'] = 0.0
    sub.to_csv(path + '/baseline1.csv', index=False, sep=',', encoding='UTF-8')


lgb_train()

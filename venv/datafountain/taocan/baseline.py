#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/10/11 下午6:48
# @Author   :hwwu
# @File     :baseline.py

import pandas as pd
import numpy as np

path = '/Users/liyangyang/Downloads/datafountain/taocan/'


###service_type,is_mix_service,online_time,1_total_fee,2_total_fee,3_total_fee,4_total_fee,
# month_traffic,many_over_bill,contract_type,contract_time,is_promise_low_consume,net_service,
# pay_times,pay_num,last_month_traffic,local_trafffic_month,local_caller_time,service1_caller_time,
# service2_caller_time,gender,age,complaint_level,former_complaint_num,former_complaint_fee,
# current_service,user_id
def getdata(data, f=True):
    # data = pd.read_csv(path + 'train_all.csv')
    if f:
        data.loc[data['current_service'] == 90063345, 'current_service'] = 0
        data.loc[data['current_service'] == 89950166, 'current_service'] = 1
        data.loc[data['current_service'] == 89950167, 'current_service'] = 2
        data.loc[data['current_service'] == 99999828, 'current_service'] = 3
        data.loc[data['current_service'] == 90109916, 'current_service'] = 4
        data.loc[data['current_service'] == 89950168, 'current_service'] = 5
        data.loc[data['current_service'] == 99999827, 'current_service'] = 6
        data.loc[data['current_service'] == 99999826, 'current_service'] = 7
        data.loc[data['current_service'] == 90155946, 'current_service'] = 8
        data.loc[data['current_service'] == 99999830, 'current_service'] = 9
        data.loc[data['current_service'] == 99999825, 'current_service'] = 10
        data.loc[data['age'] == '\\N', 'age'] = 0
        data.loc[data['gender'] == '\\N', 'gender'] = 0

    data['age'] = data['age'].astype('int64')
    data.loc[data['age'] < 20, 'age'] = 0
    data.loc[(data['age'] >= 20) & (data['age'] < 30), 'age'] = 1
    data.loc[(data['age'] >= 30) & (data['age'] < 40), 'age'] = 2
    data.loc[(data['age'] >= 40) & (data['age'] < 50), 'age'] = 3
    data.loc[data['age'] >= 50, 'age'] = 4

    data['gender'] = data['gender'].astype('int64')

    data.loc[data['2_total_fee'] == '\\N', '2_total_fee'] = 0.0
    data.loc[data['3_total_fee'] == '\\N', '3_total_fee'] = 0.0
    data['2_total_fee'] = data['2_total_fee'].astype('float64')
    data['3_total_fee'] = data['3_total_fee'].astype('float64')
    data.loc[data['1_total_fee'] > 500.0, '1_total_fee'] = 500.0
    data.loc[data['2_total_fee'] > 500.0, '2_total_fee'] = 500.0
    data.loc[data['3_total_fee'] > 500.0, '3_total_fee'] = 500.0
    data.loc[data['4_total_fee'] > 500.0, '4_total_fee'] = 500.0

    data['total_fee'] = 0
    data.loc[data['1_total_fee'] < .0, 'total_fee'] = 1
    data.loc[data['2_total_fee'] < .0, 'total_fee'] = 1
    data.loc[data['3_total_fee'] < .0, 'total_fee'] = 1
    data.loc[data['4_total_fee'] < .0, 'total_fee'] = 1
    data.loc[data['1_total_fee'] > 499.0, 'total_fee'] = 2
    data.loc[data['2_total_fee'] > 499.0, 'total_fee'] = 2
    data.loc[data['3_total_fee'] > 499.0, 'total_fee'] = 2
    data.loc[data['4_total_fee'] > 499.0, 'total_fee'] = 2

    data['month_traffic_0'] = 0
    data.loc[(data['month_traffic'] > 0) & (data['month_traffic'] < 1024), 'month_traffic_0'] = 1
    data.loc[data['month_traffic'] == 1024.0, 'month_traffic_0'] = 2
    data.loc[data['month_traffic'] > 1024, 'month_traffic_0'] = 3

    data.loc[data['online_time'] > 140, 'online_time'] = 140

    data['pay_ave'] = data['pay_num'] / data['pay_times']
    data.loc[data['pay_times'] > 10, 'pay_times'] = 10

    data['my_traffic'] = data['last_month_traffic'].apply(lambda x: parse_traffic(x))

    data = data.drop(['local_trafffic_month'], axis=1)
    data = data.drop(['last_month_traffic'], axis=1)
    data = data.drop(['month_traffic'], axis=1)

    data.loc[data['local_caller_time'] == 0.0, 'local_caller_time'] = 0
    data.loc[(data['local_caller_time'] > 0) & (data['local_caller_time'] < 10), 'local_caller_time'] = 1
    data.loc[(data['local_caller_time'] >= 10) & (data['local_caller_time'] < 100), 'local_caller_time'] = 2
    data.loc[data['local_caller_time'] >= 100, 'local_caller_time'] = 3

    data.loc[data['service1_caller_time'] == 0.0, 'service1_caller_time'] = 0
    data.loc[(data['service1_caller_time'] > 0) & (data['service1_caller_time'] < 10), 'service1_caller_time'] = 1
    data.loc[(data['service1_caller_time'] >= 10) & (data['service1_caller_time'] < 100), 'service1_caller_time'] = 2
    data.loc[data['service1_caller_time'] >= 100, 'service1_caller_time'] = 3

    data.loc[data['service2_caller_time'] == 0.0, 'service2_caller_time'] = 0
    data.loc[(data['service2_caller_time'] > 0) & (data['service2_caller_time'] < 10), 'service2_caller_time'] = 1
    data.loc[(data['service2_caller_time'] >= 10) & (data['service2_caller_time'] < 100), 'service2_caller_time'] = 2
    data.loc[data['service2_caller_time'] >= 100, 'service2_caller_time'] = 3

    data['complaint_num'] = 0
    data.loc[data['former_complaint_num'] > 0, 'complaint_num'] = 1

    data['complaint_fee'] = 0
    data.loc[data['former_complaint_fee'] > 0, 'complaint_fee'] = 1

    return data


def parse_traffic(x):
    m = x / 1024.0
    if m == 0.0:
        return 0
    elif m < 1.0:
        return 0.5
    elif m == 1.0:
        return 1
    elif m < 2.0:
        return 1.5
    elif m == 2.0:
        return 2
    elif m < 3.0:
        return 2.5
    elif m == 3.0:
        return 3
    elif m < 4.0:
        return 3.5
    elif m == 4.0:
        return 4
    else:
        return 5


data = pd.read_csv(path + 'train_all.csv')
data = getdata(data)
train_data = data[5000:]
test_data = data[:5000]

train_x = train_data.drop(['user_id', 'current_service'], axis=1)
train_y = train_data['current_service']

test_x = test_data.drop(['user_id', 'current_service'], axis=1)
test_y = test_data['current_service']
test_y = np.array(test_y)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

####### test数据
republish_test_data = pd.read_csv(path + 'republish_test.csv')
republish_test_data = getdata(republish_test_data, f=False)
print('republish_test_data: ', republish_test_data.shape)

user_id = republish_test_data['user_id']
republish_test = republish_test_data.drop(['user_id'], axis=1)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

Y_CAT = pd.Categorical(train_y)
X_train, X_test, y_train, y_test = train_test_split(train_x, Y_CAT.codes, test_size=0.05, random_state=666)

params = {
    'boosting_type': 'gbdt',
    'max_depth': -1,
    # 'metric': {'multi_logloss'},
    'num_class': 11,
    'objective': 'multiclass',
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'min_child_weight': 50,
    'reg_alpha': 0.4640,
    'reg_lambda': 0.8571,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'subsample_freq': 1,
    'n_jobs': -1,
    'random_state': 666
}


def train():
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    print('begin train')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=400,
                    verbose_eval=1)
    y_pred = gbm.predict(test_x)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    print(len(y_pred))
    count = 0
    for i in range(len(y_pred)):
        # print(test_y[i:i+1][0])
        if (y_pred[i] == test_y[i:i + 1][0]):
            # print(y_pred[i], test_y[i:i + 1][0])
            count += 1
    print(count, len(y_pred), count / len(y_pred))

    ##write result
    republish_pred = gbm.predict(republish_test)
    republish_pred = [list(x).index(max(x)) for x in republish_pred]

    republish_pred = pd.DataFrame(republish_pred)

    sub = pd.concat([user_id, republish_pred], axis=1)
    print(sub.shape)
    sub.columns = ['user_id', 'current_service']
    sub.loc[sub['current_service'] == 0, 'current_service'] = 90063345
    sub.loc[sub['current_service'] == 1, 'current_service'] = 89950166
    sub.loc[sub['current_service'] == 2, 'current_service'] = 89950167
    sub.loc[sub['current_service'] == 3, 'current_service'] = 99999828
    sub.loc[sub['current_service'] == 4, 'current_service'] = 90109916
    sub.loc[sub['current_service'] == 5, 'current_service'] = 89950168
    sub.loc[sub['current_service'] == 6, 'current_service'] = 99999827
    sub.loc[sub['current_service'] == 7, 'current_service'] = 99999826
    sub.loc[sub['current_service'] == 8, 'current_service'] = 90155946
    sub.loc[sub['current_service'] == 9, 'current_service'] = 99999830
    sub.loc[sub['current_service'] == 10, 'current_service'] = 99999825
    sub.to_csv(path + '/baseline.csv', index=False, sep=',', encoding='UTF-8')


train()

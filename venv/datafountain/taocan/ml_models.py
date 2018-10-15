#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/10/15 上午11:28
# @Author   :hwwu
# @File     :ml_models.py

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
train_data = data
train_x = train_data.drop(['user_id', 'current_service'], axis=1)
train_y = train_data['current_service']

####### test数据
republish_test_data = pd.read_csv(path + 'republish_test.csv')
republish_test_data = getdata(republish_test_data, f=False)
# print('republish_test_data: ', republish_test_data.shape)

user_id = republish_test_data['user_id']
republish_test = republish_test_data.drop(['user_id'], axis=1)

from sklearn.model_selection import train_test_split

Y_CAT = pd.Categorical(train_y)
X_train, X_test, y_train, y_test = train_test_split(train_x, Y_CAT.codes, test_size=0.05, random_state=666)

y_test = np.array(y_test)


def score(y_pred):
    y_pred = [list(x).index(max(x)) for x in y_pred]
    count = 0
    for i in range(len(y_pred)):
        # print(test_y[i:i+1][0])
        if (y_pred[i] == y_test[i:i + 1][0]):
            # print(y_pred[i], test_y[i:i + 1][0])
            count += 1
    print(count, len(y_pred), count / len(y_pred))


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# clf = MultinomialNB()
# clf.fit(X_train, y_train)
# print("多项式贝叶斯分类器20折交叉验证得分: ", np.mean(cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')))
# score(clf.predict(X_test))
#
from sklearn import svm

lin_clf = svm.LinearSVC(class_weight='balanced')
lin_clf.fit(X_train, y_train)
print("svm分类器20折交叉验证得分: ", np.mean(cross_val_score(lin_clf, X_train, y_train, cv=5, scoring='accuracy')))
score(lin_clf.predict(X_test))

from sklearn.ensemble import RandomForestClassifier

lin_forest = RandomForestClassifier(n_estimators=10, random_state=1, class_weight='balanced')
lin_forest.fit(X_train, y_train)
print("RandomForestClassifier分类器20折交叉验证得分: ",
      np.mean(cross_val_score(lin_forest, X_train, y_train, cv=5, scoring='accuracy')))
score(lin_forest.predict(X_test))

import xgboost as xgb

model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468)
model_xgb.fit(X_train, y_train)
print("model_xgb分类器20折交叉验证得分: ",
      np.mean(cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='accuracy')))
score(model_xgb.predict(X_test))

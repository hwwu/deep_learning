#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/8/7 下午1:21
# @Author   :hwwu
# @File     :dnn_model.py

import tensorflow as tf
import numpy as np
import sys

path = '/Users/liyangyang/PycharmProjects/mypy/venv/datafountain/guangfudianzhan/'
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
       7437, 11832, 16437, 15355, 3152, 3612,3611]


def load_train_data():
    train_ = read_data.read_result_data('public.train.csv')
    train_x = train_[:, 2:21]
    train_y = train_[:, 21]
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
                x.append([
                    train_x[i][1],
                    train_x[i][2],
                    train_x[i][0],
                    id,
                    train_x[i][3],
                    train_x[i][4], train_x[i][5], train_x[i][6],
                    train_x[i][7], train_x[i][8], train_x[i][9],
                    train_x[i][7] / (train_x[i][10] + 0.1), train_x[i][8] / (train_x[i][11] + 0.1),
                    train_x[i][9] / (train_x[i][12] + 0.1),
                    train_x[i][10], train_x[i][11], train_x[i][12],
                    train_x[i][13], train_x[i][14], train_x[i][15],
                    train_x[i][4] * train_x[i][13], train_x[i][5] * train_x[i][14], train_x[i][6] * train_x[i][15],
                    train_x[i][18],
                    train_x[i][17],
                    train_x[i][16]
                ])
                # x.append(train_x[i])
                y.append(abs(train_y[i]))
    print(len(x))
    # for i in  range(10):
    #     print(x[i])
    return x, y


def load_test_data():
    # train_ = read_data.read_result_data('test_data_all.csv')
    train_ = read_data.read_result_data('public.test.csv')
    train_x = train_[:, 2:21]
    train_y = train_[:, 1]

    train_len = len(train_y)
    train_y.shape = (1, train_len)
    train_y = np.transpose(train_y)

    x, y = [], []
    for i in range(train_len):
        if ((round(train_x[i][0], 2) != 0.01) | (round(train_x[i][1], 1) != 0.1)):

            id = 0.0
            for j in range(len(dis)):
                if (train_y[i] < dis[j]):
                    id = 0.5 - np.abs((int(train_y[i]) - dis[j - 1]) / (dis[j] - dis[j - 1]) - 0.5)
                    break

            if (train_y[i] not in dic):
                x.append([
                    train_x[i][1],
                    train_x[i][2],
                    train_x[i][3],
                    train_x[i][4],
                    train_x[i][0],
                    id,
                    train_x[i][5],
                    train_x[i][6],
                    # train_x[i][7],
                    # train_x[i][8],
                    # train_x[i][9],
                    train_x[i][10], train_x[i][11], train_x[i][12],
                    train_x[i][13], train_x[i][14],
                    train_x[i][15],
                    train_x[i][17],
                    train_x[i][18],
                    train_x[i][16]
                ])
                # x.append(train_x[i])
                y.append(train_y[i])
    print(len(x))
    return x, y


x, y = load_train_data()

train_x = np.reshape(x[1::1], (-1, 26))
train_y = np.reshape(y[1::1], (-1, 1))
test_x = np.reshape(x[1::1], (-1, 26))
test_y = np.reshape(y[1::1], (-1, 1))
#
# x1, y1 = load_test_data()
# test_x = np.reshape(x1, (-1, 17))
# test_y = np.reshape(y1, (-1, 1))

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=17)]
classifier = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                           hidden_units=[1],
                                           optimizer=tf.train.AdamOptimizer(
                                           learning_rate=0.0001
                                           ),
                                           activation_fn=tf.nn.leaky_relu)
# classifier = tf.contrib.learn.DNNLinearCombinedRegressor(dnn_feature_columns=feature_columns,
#                                                          dnn_hidden_units=[1],
#                                                          dnn_optimizer=tf.train.AdamOptimizer(
#                                                          learning_rate=0.001
#                                                          ))
classifier.fit(x=train_x,
               y=train_y,
               max_steps=40000)

print(classifier.evaluate(x=train_x, y=train_y))

y = classifier.predict(test_x)
y_=[]
for i in y:
    y_.append([i])

# r = []
# for i in range(8337):
#     id = test_y[i][0]
#     p = y_[i][0]
#     r.append([id, p])
# np.savetxt('/Users/liyangyang/Downloads/datafountain/guangdianfute/test_data_3', r)


error = []
for i in range(len(test_y)):
    if((test_y[i] - y_[i]) * (test_y[i] - y_[i]) > 1):
        print(test_x[i], test_y[i], y_[i])
    error.append(test_y[i] - y_[i])

squaredError = []
absError = []
for val in error:
    squaredError.append(val * val)  # target-prediction之差平方

print("Square Error: ", sorted(squaredError, reverse=True))

print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
from math import sqrt

print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE

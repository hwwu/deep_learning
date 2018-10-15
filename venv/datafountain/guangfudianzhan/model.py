#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/6/27 上午10:17
# @Author   :hwwu
# @File     :model.py

import tensorflow as tf
import matplotlib.pyplot as plt
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
       7437, 11832, 15355, 3152, 3612, 3611]


def load_train_data():
    train_ = read_data.read_result_data('public.train.csv')
    train_x = train_[:, 2:21]
    train_y = train_[:, 21]
    train_z = train_[:, 1]

    train_len = len(train_y)
    train_y.shape = (1, train_len)
    train_y = np.transpose(train_y)

    x, y, z = [], [], []
    for i in range(train_len):
        if ((round(train_x[i][0], 2) != 0.01) | (round(train_x[i][1], 1) != 0.1)):

            id = 0.0
            for j in range(len(dis)):
                if (train_z[i] < dis[j]):
                    id = 0.5 - np.abs((int(train_z[i]) - dis[j - 1]) / (dis[j] - dis[j - 1]) - 0.5)
                    break

            if (train_z[i] not in dic):
                x.append([id,
                          train_x[i][0],
                          train_x[i][2] * train_x[i][3],
                          # train_x[i][2] * train_x[i][5],
                          # train_x[i][2] * train_x[i][6],
                          # train_x[i][13],
                          # train_x[i][14],
                          train_x[i][16]
                          # train_x[i][17],
                          # train_x[i][18]
                          ])
                # x.append(train_x[i])
                y.append(train_y[i])
                # z.append([train_x[i][13] * 0.6 + train_x[i][14] * 0.15 + train_x[i][15] * 0.25])
    print(len(x))
    # for i in  range(10):
    #     print(x[i])
    return x, y, z


def load_test_data():
    train_ = read_data.read_result_data('test_data_all.csv')
    # train_ = read_data.read_result_data('public.test.csv')
    train_x = train_[:, 2:21]
    train_y = train_[:, 1]

    train_len = len(train_y)
    train_y.shape = (1, train_len)
    train_y = np.transpose(train_y)

    x, y, z = [], [], []
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
                    train_x[i][0],
                    id,
                    train_x[i][3],
                    # train_x[i][4], train_x[i][5], train_x[i][6],
                    # train_x[i][7] / train_x[i][10], train_x[i][8] / train_x[i][10], train_x[i][9] / train_x[i][10]
                    # train_x[i][10], train_x[i][11], train_x[i][12],
                    # train_x[i][13], train_x[i][14], train_x[i][15],
                    # train_x[i][4] * train_x[i][13]/1000, train_x[i][5] * train_x[i][14]/1000, train_x[i][6] * train_x[i][15]/1000,
                    # train_x[i][18],
                    train_x[i][17]
                ])
                # x.append(train_x[i])
                y.append(train_y[i])
                z.append([train_x[i][13] * 0.6 + train_x[i][14] * 0.15 + train_x[i][15] * 0.25])
    print(len(x))
    return x, y, z


hidden = 4

weights = {
    'in': tf.Variable(tf.random_normal([hidden, hidden])),
    'mid1': tf.Variable(tf.random_normal([hidden, hidden])),
    'mid2': tf.Variable(tf.random_normal([hidden, hidden])),
    'out': tf.Variable(tf.random_normal([hidden, 1]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[hidden, ])),
    'mid': tf.Variable(tf.constant(0.1, shape=[hidden, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

tf_x = tf.placeholder("float32", shape=[None, hidden])
tf_y = tf.placeholder("float32", shape=[None, 1])
# tf_z = tf.placeholder("float32", shape=[None, 1])

global_step = tf.placeholder(tf.int32)
# learning_rate = tf.train.exponential_decay(1e-3, global_step, 10000, 0.1, staircase=True)
learning_rate = 1e-4

# 第一层
# weightIn = weights['in']
# biasesIn = biases['in']
# # tf_x = tf.nn.dropout(tf_x, 0.5)
# input = tf.reshape(tf_x, [-1, 4])
# inputRnn = tf.matmul(tf_x, weightIn) + biasesIn
# 激活层
# inputRnn = tf.nn.leaky_relu(inputRnn)
#
# # 第二层
# weightMid1 = weights['mid1']
# biasesMid1 = biases['in']
# inputRnn = tf.matmul(inputRnn, weightMid1) + biasesMid1
# # 激活层
# inputRnn = tf.nn.leaky_relu(inputRnn)
#
# #
# # # 第三层
# weightMid2 = weights['mid1']
# biasesMid2 = biases['in']
# inputRnn = tf.matmul(inputRnn, weightMid2) + biasesMid2
# # 激活层
# inputRnn = tf.nn.leaky_relu(inputRnn)
#
#
# # BN归一化层+激活层
# # https://blog.csdn.net/lanchunhui/article/details/70792458 这边文章解释的比较清楚
# batch_mean, batch_var = tf.nn.moments(inputRnn, [0], keep_dims=True)
# shift = tf.Variable(tf.zeros([4]))
# scale = tf.Variable(tf.ones([1]))
# epsilon = 1e-4
# inputRnn = tf.nn.batch_normalization(inputRnn, batch_mean, batch_var, shift, scale, epsilon)
# # inputRnn = tf.nn.tanh(inputRnn)
# inputRnn = tf.nn.dropout(inputRnn, 0.8)
#
# #
# # 第四层
# weightMid3 = weights['mid1']
# biasesMid3 = biases['in']
# inputRnn = tf.matmul(inputRnn, weightMid3) + biasesMid3
# # 激活层
# inputRnn = tf.nn.leaky_relu(inputRnn)
#
# #
# # # 第五层
# weightMid4 = weights['mid2']
# biasesMid4 = biases['mid']
# inputRnn = tf.matmul(inputRnn, weightMid4) + biasesMid4
# # 激活层
# inputRnn = tf.nn.leaky_relu(inputRnn)


# 输出层
w_out = weights['out']
b_out = biases['out']
b_out_1 = biases['out']
# pred = tf.matmul(inputRnn, w_out) + b_out

pred = tf.matmul(tf_x, w_out) + b_out
# pred = tf_z / 500 + out
# pred = tf.abs(tf.matmul(inputRnn, w_out) + b_out)
#
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, weightIn)
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, weightMid1)
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, weightMid2)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_out)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.9 / 8905)
reg_term = tf.contrib.layers.apply_regularization(regularizer)

loss_ = tf.sqrt(tf.losses.mean_squared_error(tf_y, pred))
loss = loss_ + reg_term  # compute cost
# loss = tf.sqrt(tf.losses.mean_squared_error(tf_y, pred))  # compute cost
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, rmse_loss)
# loss = tf.add_n(tf.get_collection(tf.GraphKeys.WEIGHTS))
# loss = tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(tf_y, [-1]))))
# loss = tf.reduce_mean(tf.square(tf.reshape(pred, [batch_size,1]) - tf.reshape(tf_y, [batch_size,1])))
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

train_x, train_y, train_z = load_train_data()

sess = tf.Session()  # control training and others
sess.run(tf.global_variables_initializer())  # initialize var in graph
saver = tf.train.Saver(tf.global_variables())

# saver.restore(sess, '/Users/liyangyang/PycharmProjects/mypy/venv/datafountain/guangfudianzhan/model/stock.train.model')


def train():
    training_losses = []
    min = 0.1657034
    for i in range(300000):
        start = 0000
        batch = 8000
        end = start + batch
        _, loss_train = sess.run([train_op, loss],
                                 feed_dict={
                                     # tf_x: train_x[start:end],
                                     # tf_y: train_y[start:end],
                                     # tf_x: train_x,
                                     # tf_y: train_y,
                                     tf_x: train_x[i%1:end:1],
                                     tf_y: train_y[i%1:end:1],
                                     # tf_z: train_z[start:end:2],
                                     global_step: i
                                 })
        if (i % 500 == 10):
            training_losses.append(loss_train)
            saver.save(sess,
                       '/Users/liyangyang/PycharmProjects/mypy/venv/datafountain/guangfudianzhan/model/stock.train.model')
            print("step %d, train accuracy %g" % (i, loss_train))
            ls = test(i % 500)
            if (ls < min):
                min = ls
                print('save model success')
                saver.save(sess,
                           '/Users/liyangyang/PycharmProjects/mypy/venv/datafountain/guangfudianzhan/model/stock.train.min.model')

    plt.plot(training_losses)
    plt.show()


t_train_x, t_train_y, t_train_z = load_train_data()


#
#
# t_train_x, t_train_y,t_train_z = load_test_data()


def test(t):
    # start = 0
    # batch = 17243
    start = 8000
    batch = 905
    # import random
    # start = random.randint(0, 8800)
    # batch = random.randint(50, 100)
    end = start + batch
    train_test_x = t_train_x[start:end:1]
    train_test_y = t_train_y[start:end:1]
    train_test_z = t_train_z[start:end:1]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,
                      '/Users/liyangyang/PycharmProjects/mypy/venv/datafountain/guangfudianzhan/model/stock.train.model')
        res_ypred, loss = sess.run([pred, loss_],
                                   feed_dict={tf_x: train_test_x, tf_y: train_test_y
                                                # ,tf_z: train_test_z
                                              })  # 只能预测一批样本，不能预测一个样本
    print('test loss is :', loss)

    # if (t == 10):
    #     s = 0
    #     b = batch
    #     x = [i for i in range(s, b)]
    #     # 以折线图表示结果
    #     plt.figure()
    #     plt.plot(x, res_ypred[s:b], color='r', label='yuce')
    #     plt.plot(x, train_test_y[s:b], color='y', label='shiji')
    #     plt.xlabel("Time(s)")  # X轴标签
    #     plt.ylabel("Value")  # Y轴标签
    #     plt.show()
    #     print(start, batch)
    return loss

    # return train_test_y, res_ypred


train()

# train_test_y, res_ypred = test(0)
# r = []
# # for i in range(8338):
# for i in range(17243):
#     id = train_test_y[i][0]
#     p = res_ypred[i][0]
#     r.append([id, p])
# # np.savetxt('/Users/liyangyang/Downloads/datafountain/guangdianfute/test_data_3_model', r)
# np.savetxt('/Users/liyangyang/Downloads/datafountain/guangdianfute/test_data_all_1_model', r)


# train_test_y, res_ypred = test(0)
# error = []
# for i in range(len(res_ypred)):
#     if ((train_test_y[i] - res_ypred[i])*(train_test_y[i] - res_ypred[i])>4):
#         print(train_test_y[i],res_ypred[i])
#     error.append(train_test_y[i] - res_ypred[i])
#
# squaredError = []
# absError = []
# for val in error:
#     squaredError.append(val * val)  # target-prediction之差平方
#
# print("Square Error: ", sorted(squaredError,reverse=True))
#
# print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
# from math import sqrt
# print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE

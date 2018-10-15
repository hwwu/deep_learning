#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/7/16 上午11:32
# @Author   :hwwu
# @File     :read_data.py

import pandas as pd
import numpy as np

path = '/Users/liyangyang/Downloads/datafountain/guangdianfute/'


def read_original_data(file):
    data = pd.read_csv(path + file)
    data = data[(data['平均功率'] < 10000.0)]
    data = data[(data['现场温度'] > -1000.0)]
    # data = data[(data['转换效率'] < 3000.0)]
    data = data[~((data['板温']==0.01)&(data['现场温度']==0.1))]

    return data


def load_original_data(file='public.train.csv'):
    train = read_original_data(file)
    return train.reset_index()


def read_result_data(file='public.train.csv'):
    train = load_original_data(file)
    result = np.array(train)
    print(result.shape)
    return result


def write_test_result1():
    train_ = read_result_data('public.test.csv')
    train_x = train_[:, 2:21]
    train_y = train_[:, 1]

    res = []

    train_len = len(train_y)
    train_y.shape = (1, train_len)
    train_y = np.transpose(train_y)

    for i in range(train_len):
        if ((round(train_x[i][0], 2) == 0.01) & (round(train_x[i][1], 1) == 0.1)):
            res.append([train_y[i], 0.379993053])

    print(len(res))
    np.savetxt(path + 'test_data_1', res)


def write_test_result():
    train_1 = read_result_data('public.test.csv')
    train_2 = read_result_data('public.train.csv')
    train_x = train_1[:, 1:21]
    train_y = train_2[:, 1:21]

    train = np.vstack([train_x, train_y])

    train_a = train[:, ::-1].T
    train_a2 = np.lexsort(train_a)
    train = train[train_a2]

    np.savetxt(path + 'test_data_all.csv', train, fmt="%.2f", delimiter=',')


# write_test_result()


def write_result():
    x1 = np.loadtxt(path + 'test_data_1')
    x2 = np.loadtxt(path + 'test_data_3')

    user_id = []
    price = []
    for i in range(len(x1)):
        user_id.append(int(x1[i][0]))
        # price.append(round(x1[i][1],1))
        price.append(round(x1[i][1], 7))
    for i in range(len(x2)):
        user_id.append(int(x2[i][0]))
        price.append(round(x2[i][1], 7))
    english_column = pd.Series(user_id)
    number_column = pd.Series(price)
    predictions = pd.concat([english_column, number_column], axis=1)
    # another way to handle
    # save = pd.DataFrame({'user_id': user_id, 'prediction_pay_price': price})
    predictions.to_csv(path + 'result_data.csv', index=0, sep=',')


# write_result()

def write_result2():
    t = read_result_data('public.test.csv')
    x1 = np.loadtxt(path + 'test_data_all_2')
    x2 = np.loadtxt(path + 'test_data_3')
    t1 = t[:, 1]
    map = {}
    r = []
    map2 = {}
    for i in range(len(x2)):
        map2[int(x2[i][0])] = x2[i][1]
    for i in range(8409):
        if ((round(t[i][0], 2) != 0.01) | (round(t[i][1], 1) != 0.1)):
            map[int(t1[i])] = 0
    for i in range(len(x1)):
        a1 = int(x1[i][0])
        a2 = x1[i][2]
        if (a1==16437):
            r.append([a1,9.911484700000000814e+00])
            print(a1)
        elif (a1 in map2.keys()):
            r.append([a1, map2[a1]])
        elif (a1 in map.keys()):
            r.append([a1, a2])
        # else:
        #     r.append([a1,a2])

    np.savetxt('/Users/liyangyang/Downloads/datafountain/guangdianfute/test_data_2', r)


# write_result2()


def mid_merge_r():
    x1 = np.loadtxt(path + 'test_data_all_1')
    t = read_result_data('public.train.csv')
    t1 = t[:, 1]
    t2 = t[:, 21]
    map = {}
    r = []
    for i in range(9000):
        if ((round(t[i][0], 2) != 0.01) | (round(t[i][1], 1) != 0.1)):
            map[int(t1[i])] = t2[i]
    for i in range(len(x1)):
        a1 = int(x1[i][0])
        a2 = x1[i][1]
        if (a1 in map.keys()):
            r.append([a1, a2, round(map[a1], 7)])
        else:
            r.append([a1, a2, 0.0])

    np.savetxt('/Users/liyangyang/Downloads/datafountain/guangdianfute/test_data_all_2', r)


# mid_merge_r()

import matplotlib.pyplot as plt


def plot():
    x1 = np.loadtxt(path + 'test_data_all_2')
    s = 90*100
    b = 100
    e = s + b
    x = [i for i in range(s, e)]
    # 以折线图表示结果
    plt.figure()
    plt.plot(x, x1[s:e,1], color='r', label='yuce')
    plt.plot(x, x1[s:e, 2], color='y', label='shiji')
    plt.xlabel("Time(s)")  # X轴标签
    plt.ylabel("Value")  # Y轴标签
    plt.show()

# plot()

if __name__ == '__main__':
    pass
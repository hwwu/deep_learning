#!/usr/bin/env python3
# --coding:utf8 --
# @TIME     :2018/8/1 上午10:28
# @Author   :hwwu
# @File     :draw.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


path = '/Users/liyangyang/Downloads/datafountain/guangdianfute/'


dic = [22, 135, 591, 592, 593, 594, 595, 737, 948, 1070, 1173, 1175, 1286,
       1362, 1451, 1519, 1565, 1666, 1717, 1894, 2137, 2223, 2271, 2414,
       2579, 2797, 2875, 2916, 2986, 2684, 3723, 3597, 3599, 3603, 3605,
       3607, 3610, 3601, 3602, 3421, 3393, 3538, 3539, 3540, 5521, 6016,
       7437, 11832, 16437, 15355, 3152, 3612, 3611]

# 板温  现场温度  光照强度   转换效率   转换效率A  转换效率B   转换效率C  电压A  电压B 电压C
# 电流A   电流B   电流C      功率A     功率B      功率C     平均功率   风速   风向       发电量
def draw_data(file='public.train.csv'):
    data = pd.read_csv(path + file)
    print(data.std())
    # data = data[(data['平均功率'] < 10000.0)]
    # data = data[(data['现场温度'] > -1000.0)]
    # data = data[(data['转换效率'] < 2000.0)]
    data = data[~data['ID'].isin(dic)]
    # data = data[(data['电流A'] > 200.0)]
    # print(len(data))
    # plt.hist(data['风向'])
    # 板温 光照强度
    # xs = data['电压C']/data['电流C']
    xs = (data['光照强度']*data['转换效率'])/100/12.5
    # xs = data['现场温度']
    ys = data['发电量']
    plt.scatter(xs, ys)
    # x = [i for i in range(100)]
    # for i in range(1,11):
    #     strat=4000+i*100
    #     plt.plot(x, (data['平均功率']/1000*2)[strat:strat+100], color='r', label='yuce')
    #     plt.plot(x, data['发电量'][strat:strat+100], color='y', label='shiji')
    #     plt.show()
    # strat = 2200
    # plt.plot(x, (data['平均功率'] / 1000 * 2)[strat:strat + 200], color='r', label='yuce')
    # plt.plot(x, data['发电量'][strat:strat + 200], color='y', label='shiji')
    plt.show()
    # print(data.head())

import seaborn as sns
def neighborhood(file='public.train.csv'):
    train = pd.read_csv(path + file)
    train.rename(columns={
        '板温': 'a', '现场温度': 'b', '光照强度': 'c', '转换效率': 'd', '转换效率A': 'e',
        '转换效率B': 'f', '转换效率C': 'g', '电压A': 'h', '电压B': 'i', '电压C': 'j',
        '电流A': 'k', '电流B': 'l', '电流C': 'm', '功率A': 'n', '功率B': 'o',
        '功率C': 'p', '平均功率': 'q', '风速': 'r', '风向': 's', '发电量': 't'
                          }, inplace=True)
    k = 10  # number of variables for heatmap
    corrmat = train.corr()
    cols = corrmat.nlargest(k, 't')['t'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()
# neighborhood()

file='public.train.csv'
def t():
    train = pd.read_csv(path + file)
    y = train['发电量']
    train_labels = y.values.copy
    print(y.describe())
    sns.distplot(y)
    print('Skewness: %f' % y.skew())
    print('Kurtosis: %f' % y.kurt())
    # 得到训练集的数值特征和类别特征

    from scipy.stats import skew
    # log transform the target use log(1+x)
    train["发电量"] = np.log1p(train["发电量"])
    sns.distplot(train['发电量'])
    print("Skewness: %f" % train['发电量'].skew())
    print("Kurtosis: %f" % train['发电量'].kurt())
# t()

draw_data()
# draw_data('public.test.csv')


from sklearn.feature_selection import SelectKBest
import sklearn


import sys

path = '/Users/liyangyang/PycharmProjects/mypy/venv/datafountain/guangfudianzhan/'
sys.path.append(path)
import read_data

dis = [1,190,379,567,755,940,1123,1314,1503,1505,1694,1879,
2070,2257,2444,2632,2823,3013,3202,3379,3567,3746,3927,4089,
4278,4459,4648,4652,4821,5010,5013,5017,5059,5061,5069,5074,
5077,5281,5285,5287,5292,5508,5703,5911,5913,5916,5918,6121,
6337,6524,6528,6531,6534,6723,6923,7116,7326,7535,7740,7937,
8146,8245,8258,8310,8488,8705,8711,8878,9088,9296,9505,9719,
9916,10124,10335,10544,10736,10914,10917,11119,11331,11540,
11753,11963,12170,12381,12592,12802,13009,13214,13426,13617,
13830,14032,14243,14457,14666,14882,15091,15299,15508,15719,
15937,16144,16348,16540,16747,16925,17133,17342,
17527,17543,17745,17876]


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
                if (train_z[i]<dis[j]):
                    id = 0.5 - np.abs((int(train_z[i]) - dis[j - 1]) / (dis[j] - dis[j - 1])-0.5)
                    break


            x.append([
                      id,
                      train_x[i][0], train_x[i][1],
                      train_x[i][2],
                      train_x[i][3],
                      train_x[i][4],train_x[i][5], train_x[i][6],
                      train_x[i][7],train_x[i][8], train_x[i][9],
                      train_x[i][10],train_x[i][11], train_x[i][12],
                      train_x[i][13], train_x[i][14], train_x[i][15], train_x[i][16],
                      train_x[i][17],train_x[i][18]
            ])
            # x.append(train_x[i])
            y.append(train_y[i])

    return x,y

# x,y = load_train_data()
# model1 = SelectKBest(sklearn.feature_selection.f_regression, k=5)#选择k个最佳特征
# r = model1.fit_transform(x, y)#iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征
# print(r)
# print(model1.scores_)
# print(model1.pvalues_)
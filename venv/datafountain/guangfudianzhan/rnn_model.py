#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/6/21 下午1:27
# @Author   :hwwu
# @File     :PricePredictor.py

import codecs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd


class PricePredictor:
    # lstm param
    timeStep = 19
    hiddenUnitSize = 38  # 隐藏层神经元数量
    batchSize = 88  # 每一批次训练多少个样例
    inputSize = 19  # 输入维度
    outputSize = 1  # 输出维度
    lr = 0.0001  # 学习率
    train_x, train_y = [], []  # 训练数据集
    dataFile = '/Users/liyangyang/Downloads/datafountain/guangdianfute/public.train.csv'
    testFile = '/Users/liyangyang/Downloads/datafountain/guangdianfute/public.test.csv'
    train_data = []
    X = tf.placeholder(tf.float32, [None, timeStep, inputSize])
    Y = tf.placeholder(tf.float32, [None, timeStep])
    # Y = tf.placeholder(tf.float32, [None, timeStep, outputSize])
    weights = {
        'in': tf.Variable(tf.random_normal([inputSize, hiddenUnitSize])),
        'out': tf.Variable(tf.random_normal([hiddenUnitSize, 1]))
    }

    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[hiddenUnitSize, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

    savePath = '/Users/liyangyang/PycharmProjects/mypy/venv/datafountain/guangfudianzhan/model/stock.train.model'

    def loadData(self):
        data = pd.read_csv(self.dataFile)
        data = np.array(data)
        train_len = len(data)
        train = []
        for i in range(train_len):
            if ((round(data[i][1], 2) != 0.01) | (round(data[i][2], 1) != 0.1)):
                if (data[i][2] < -1000):
                    print(data[i][2])
                    data[i][2] = -6.0
                if (data[i][19] > 360):
                    data[i][19] -= 360
                if (data[i][20] < 0):
                    data[i][20] = -data[i][20]
                train.append(data[i])
        print(len(train))
        self.train_data = np.array(train)

    # 构造数据
    def buildTrainDataSet(self):
        x_ = self.train_data[:, 1:20]
        y_ = self.train_data[:, 20]
        for i in range(len(self.train_data) - self.timeStep - 1):
            x = x_[i:i + self.timeStep]
            y = y_[i:i + self.timeStep]
            self.train_x.append(x.tolist())
            self.train_y.append(y.tolist())

    # lstm算法定义
    def lstm(self, batchSize=None):
        if batchSize is None:
            batchSize = self.batchSize
        weightIn = self.weights['in']
        biasesIn = self.biases['in']
        input = tf.reshape(self.X, [-1, self.inputSize])
        inputRnn = tf.matmul(input, weightIn) + biasesIn
        inputRnn = tf.reshape(inputRnn, [-1, self.timeStep, self.hiddenUnitSize])  # 将tensor转成3维，作为lstm cell的输入
        # cell=tf.nn.rnn_cell.BasicLSTMCell(self.hiddenUnitSize, reuse=True)
        # initState=cell.zero_state(batchSize,dtype=tf.float32)
        # output_rnn,final_states=tf.nn.dynamic_rnn(cell, inputRnn,initial_state=initState, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果

        # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenUnitSize, forget_bias=1.0, state_is_tuple=True)
        # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob

        # 运行test的时候注释掉这段，不能dropout
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 5, state_is_tuple=True)
        # **步骤5：用全零来初始化state
        init_state = mlstm_cell.zero_state(batchSize, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(mlstm_cell, inputRnn, initial_state=init_state,
                                                     dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果

        output = tf.reshape(output_rnn, [-1, self.hiddenUnitSize])  # 作为输出层的输入
        w_out = self.weights['out']
        b_out = self.biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

    # 训练模型
    def trainLstm(self):
        pred, _ = self.lstm()
        # 定义损失函数
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(self.Y, [-1]))))
        # 定义训练模型
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess,self.savePath)
            # 重复训练100次，训练是一个耗时的过程
            for i in range(1000):
                step = 0
                start = 0
                end = start + self.batchSize
                while end < len(self.train_x):
                    _, loss_ = sess.run([train_op, loss], feed_dict={self.X: self.train_x[start:end],
                                                                                 self.Y: self.train_y[start:end]})
                    # start += 1
                    start += self.batchSize
                    end = start + self.batchSize
                    # 每10步保存一次参数
                    if step % 500 == 0:
                        print('test loss is :', i, loss_)
                    if (i % 10 == 0) & (step % 500 == 0):
                        print("保存模型")
                        saver.save(sess, self.savePath)
                    step += 1

    def prediction(self):
        pred, _ = self.lstm()  # 预测时只输入[1,time_step,inputSize]的测试数据
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            saver.restore(sess, self.savePath)
            # 取训练集最后一行为测试样本. shape=[1,time_step,inputSize]
            result = []
            start = 20
            end = start + self.batchSize
            # while end < len(self.train_x):
            pred = sess.run([pred], feed_dict={self.X: self.train_x[start:end]
                                                            })
            # 以折线图表示结果
            p = np.reshape(pred, [self.batchSize, -1])
            s = 0
            b = self.timeStep
            x = [i for i in range(s, b*19)]
            # 以折线图表示结果
            plt.figure()
            plt.plot(x, p[0], color='r', label='yuce')
            plt.plot(x, self.train_y[s:b], color='y', label='shiji')
            plt.xlabel("Time(s)")  # X轴标签
            plt.ylabel("Value")  # Y轴标签
            plt.show()


predictor = PricePredictor()
predictor.loadData()

# 构建训练数据
predictor.buildTrainDataSet()

# # 模型训练
predictor.trainLstm()
#
# # 预测－预测前需要先完成模型训练
# predictor.prediction()

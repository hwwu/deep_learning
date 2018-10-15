#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/10/12 下午3:56
# @Author   :hwwu
# @File     :tensorflow_modle.py

import tensorflow as tf
import os
import pandas as pd
import numpy as np

model_path = os.getcwd() + '/tf_model/'
model_file = model_path + 'stock.model'

path = '/Users/liyangyang/Downloads/datafountain/taocan/'


###service_type,is_mix_service,online_time,1_total_fee,2_total_fee,3_total_fee,4_total_fee,
# month_traffic,many_over_bill,contract_type,contract_time,is_promise_low_consume,net_service,
# pay_times,pay_num,last_month_traffic,local_trafffic_month,local_caller_time,service1_caller_time,
# service2_caller_time,gender,age,complaint_level,former_complaint_num,former_complaint_fee,
# current_service,user_id
def getdata():
    data = pd.read_csv(path + 'train_all.csv')
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
    data['age'] = data['age'].astype('int64')
    data.loc[data['age'] < 20, 'age'] = 0
    data.loc[(data['age'] >= 20) & (data['age'] < 30), 'age'] = 1
    data.loc[(data['age'] >= 30) & (data['age'] < 40), 'age'] = 2
    data.loc[(data['age'] >= 40) & (data['age'] < 50), 'age'] = 3
    data.loc[data['age'] >= 50, 'age'] = 4

    data.loc[data['gender'] == '\\N', 'gender'] = 0
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

    ##tf 需要处理超大的值 否则不会收敛
    data = data.drop(['former_complaint_num'], axis=1)
    data = data.drop(['former_complaint_fee'], axis=1)
    data.loc[data['1_total_fee'] < .0, '1_total_fee'] = -0.1
    data.loc[data['2_total_fee'] < .0, '2_total_fee'] = -0.1
    data.loc[data['3_total_fee'] < .0, '3_total_fee'] = -0.1
    data.loc[data['4_total_fee'] < .0, '4_total_fee'] = -0.1

    data.loc[data['pay_num'] > 500, 'pay_num'] = 500

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


data = getdata()
train_data = data[:700000]
test_data = data[700000:]

train_x = train_data.drop(['user_id', 'current_service'], axis=1)
train_y = train_data['current_service']

test_x = test_data.drop(['user_id', 'current_service'], axis=1)
test_y = test_data['current_service']
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

weights = {
    'in': tf.Variable(tf.random_normal([26, 52])),
    'mid': tf.Variable(tf.random_normal([52, 52])),
    'out': tf.Variable(tf.random_normal([52, 11]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[52, ])),
    'mid': tf.Variable(tf.constant(0.1, shape=[52, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[11, ]))
}

tf_x = tf.placeholder("float32", shape=[None, 26])
tf_y = tf.placeholder("int64", shape=[None])

global_step = tf.placeholder(tf.int32)
learning_rate = 0.001

inputRnn = tf_x
weightIn = weights['in']
biasesIn = biases['in']
inputRnn = tf.matmul(inputRnn, weightIn) + biasesIn
inputRnn = tf.nn.leaky_relu(inputRnn)
inputRnn = tf.nn.dropout(inputRnn, 0.5)
#
# 第二层
weightMid1 = weights['mid']
biasesMid1 = biases['mid']
inputRnn = tf.matmul(inputRnn, weightMid1) + biasesMid1
# 激活层
inputRnn = tf.nn.leaky_relu(inputRnn)
# #
# # # 第三层
# # weightMid2 = weights['mid']
# # biasesMid2 = biases['mid']
# # inputRnn = tf.matmul(inputRnn, weightMid2) + biasesMid2
# # 激活层
# # inputRnn = tf.nn.tanh(inputRnn)
w_out = weights['out']
b_out = biases['out']

tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(weightIn))
tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(weightMid1))
tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(w_out))
# inputRnn = tf.nn.dropout(inputRnn, 0.5)
p = tf.matmul(inputRnn, w_out) + b_out
pred = tf.nn.softmax(p)
print(pred.shape)
# pred = tf.argmax(p, 1)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_y,
                                                        logits=p)  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
losses = tf.reduce_mean(losses)
loss = losses + tf.add_n(tf.get_collection("losses"))
# loss = losses
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

# train_x, train_y = load_train_data()
sess = tf.Session()  # control training and others
saver = tf.train.Saver()
if os.path.exists(model_path):
    print("Restoring Variables from model.")
    saver.restore(sess, model_file)
else:
    print('Initializing Variables')
    sess.run(tf.global_variables_initializer())


def train():
    training_losses = []
    number_of_training_data = len(train_x)
    data_x = train_x
    data_y = train_y
    mi = 0.79370314
    for i in range(10000):
        batch_size = 5000
        loss1, acc, counter = 0.0, 0.0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),
                              range(batch_size, number_of_training_data, batch_size)):
            counter = counter + 1
            _, loss_train, accuracy_train = sess.run([train_op, loss, accuracy],
                                                     feed_dict={tf_x: data_x[start:end],
                                                                tf_y: data_y[start:end],
                                                                global_step: i
                                                                })
            loss1, acc = loss1 + loss_train, acc + accuracy_train
            if counter % 10 == 0:
                training_losses.append(loss_train)
                print("step %d, Batch %d ,train loss %g, train accuracy %g" % (
                    i, counter, loss1 / float(counter), acc / float(counter)))
                if counter % 100 == 0:
                    saver.save(sess, model_file)
                    score = test()
                    if score > mi:
                        print('save best model..')
                        saver.save(sess, model_file + '.max')
                        mi = score


def test():
    train_test_x = test_x
    train_test_y = test_y
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_file)
        loss_, accuracy_, pred_ = sess.run([loss, accuracy, pred],
                                           feed_dict={tf_x: train_test_x, tf_y: train_test_y})  # 只能预测一批样本，不能预测一个样本
    print('test accuracy_ is ', accuracy_)
    return accuracy_


train()

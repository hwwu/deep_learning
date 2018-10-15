#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/8/27 下午4:28
# @Author   :hwwu
# @File     :p4_cnn_sentence_classification.py

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import pandas as pd
import time


import sys

sys.path.append("/Users/liyangyang/PycharmProjects/mypy/venv/dwb/testcnn")
import data_helpers
from tensorflow.contrib import learn

vocab_processor_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/testcnn/vocab'



def model():
    print("started...")
    print("Loading data...")

    x, y = data_helpers.load_data_and_labels('train_set.csv')

    # vocab_processor = learn.preprocessing.VocabularyProcessor(2000,min_frequency=2)
    # x = np.array(list(vocab_processor.fit_transform(x_train)))
    # vocab_processor.save(vocab_processor_path)

    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_processor_path)
    x = np.array(list(vocab_processor.transform(x)))

    trainX = x[20000:100000]
    testX = x[:20000]
    trainY = y[20000:100000]
    testY = y[:20000]
    vocab_size = len(vocab_processor.vocabulary_)

    print('trainX.shape', np.array(trainX).shape)
    print('trainY.shape', np.array(trainY).shape)
    print("testX.shape:", np.array(testX).shape)  # 2500个list.每个list代表一句话
    print("testY.shape:", np.array(testY).shape)  # 2500个label
    print("testX[0]:", testX[0])  # [17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
    # print("testY[0]:",testY[0]) #0

    # 2.Data preprocessing
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=19)  # y as one hot
    testY = to_categorical(testY, nb_classes=19)  # y as one hot

    # 3.Building convolutional network
    # (shape=None, placeholder=None, dtype=tf.float32,data_preprocessing=None, data_augmentation=None,name="InputData")
    network = input_data(shape=[None, 2000],
                         name='input')  # [None, 100] `input_data` is used as a data entry (placeholder) of a network. This placeholder will be feeded with data when training
    network = tflearn.embedding(network, input_dim=345325,
                                output_dim=128)  # [None, 100,128].embedding layer for a sequence of ids. network: Incoming 2-D Tensor. input_dim: vocabulary size, oput_dim:embedding size
    # conv_1d(incoming,nb_filter,filter_size)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu',
                      regularizer="L2")  # [batch_size, new steps1, nb_filters]. padding:"VALID",only ever drops the right-most columns
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu',
                      regularizer="L2")  # [batch_size, new steps2, nb_filters]
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu',
                      regularizer="L2")  # [batch_size, new steps3, nb_filters]
    network = merge([branch1, branch2, branch3], mode='concat',
                    axis=1)  # merge a list of `Tensor` into a single one.===>[batch_size, new steps1+new step2+new step3, nb_filters]
    network = tf.expand_dims(network,
                             2)  # [batch_size, new steps1+new step2+new step3,1, nb_filters] Inserts a dimension of 1 into a tensor's shape
    network = global_max_pool(network)  # [batch_size, pooled dim]
    network = dropout(network, 0.5)  # [batch_size, pooled dim]
    network = fully_connected(network, 19,
                              activation='softmax')  # matmul([batch_size, pooled_dim],[pooled_dim,2])---->[batch_size,2]
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/fasttext/'

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load(model_path + 'cnn_model')

    print('start model fit')
    model.fit(trainX, trainY, n_epoch=1, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=128)
    print('finashed model fit')

    model.save(model_path + 'cnn_model')
    y_ = model.predict(x[100000:])

    n = 0
    pred = np.argmax(y_,axis=1).reshape([2277,1])
    y_pred = np.reshape(y[100000:],[2277,1])
    for i in range(2277):
        print(pred[i],y_pred[i])
        if (pred[i]==y_pred[i]):
            n+=1
    print('n',n)
    print("ended...")


path = '/Users/liyangyang/Downloads/dwb/new_data/'

def get_test_words():
    print('start read test set data')
    test = pd.read_csv(path + 'test_set.csv')
    all_words = np.array(test['word_seg'])
    id = np.array(test['id'])
    print('read test set data done')
    return id, all_words


def write_result(id, predictions,b):
    r_id = []
    r_predictions = []
    for i in range(len(id)):
        r_id.append(int(id[i]))
        r_predictions.append((int(tostr(predictions[i]))+1))

    english_column = pd.Series(r_id, name='id')
    number_column = pd.Series(r_predictions, name='class')
    predictions = pd.concat([english_column, number_column], axis=1)
    # another way to handle
    # save = pd.DataFrame({'user_id': user_id, 'prediction_pay_price': price})
    name = str(b)+'_p4_cnn_result_data.csv'
    predictions.to_csv(path + name, index=0, sep=',', columns=['id', 'class'])


def tostr(s):
    s = str(s).replace('__myprefix__', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('\'', '')
    return s

def predict():
    network = input_data(shape=[None, 2000],
                         name='input')  # [None, 100] `input_data` is used as a data entry (placeholder) of a network. This placeholder will be feeded with data when training
    network = tflearn.embedding(network, input_dim=345325,
                                output_dim=128)  # [None, 100,128].embedding layer for a sequence of ids. network: Incoming 2-D Tensor. input_dim: vocabulary size, oput_dim:embedding size
    # conv_1d(incoming,nb_filter,filter_size)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu',
                      regularizer="L2")  # [batch_size, new steps1, nb_filters]. padding:"VALID",only ever drops the right-most columns
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu',
                      regularizer="L2")  # [batch_size, new steps2, nb_filters]
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu',
                      regularizer="L2")  # [batch_size, new steps3, nb_filters]
    network = merge([branch1, branch2, branch3], mode='concat',
                    axis=1)  # merge a list of `Tensor` into a single one.===>[batch_size, new steps1+new step2+new step3, nb_filters]
    network = tf.expand_dims(network,
                             2)  # [batch_size, new steps1+new step2+new step3,1, nb_filters] Inserts a dimension of 1 into a tensor's shape
    network = global_max_pool(network)  # [batch_size, pooled dim]
    network = dropout(network, 0.5)  # [batch_size, pooled dim]
    network = fully_connected(network, 19,
                              activation='softmax')  # matmul([batch_size, pooled_dim],[pooled_dim,2])---->[batch_size,2]
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/fasttext/'

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load(model_path + 'cnn_model')

    t_id, t_all_words = get_test_words()
    l = len(t_id)
    print(t_all_words.shape,t_id.shape)
    start = 99000
    end = start+3277
    n = 33
    # while(end<l):
    id = t_id[start:end]
    all_words = t_all_words[start:end]
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_processor_path)
    all_words = np.array(list(vocab_processor.transform(all_words)))
    print(all_words.shape)
    print(all_words[0])
    print('start predict data')
    predictions = model.predict(all_words)
    print('predict data done')
    pred = np.argmax(predictions, axis=1).reshape([len(all_words), 1])
    id = np.reshape(id, [len(all_words), 1])
    write_result(id, pred,n)
    print(n,start,end)
        # n+=1
        # start+=3000
        # end+=3000
        # time.sleep(120)

# predict()

# def write_result1():
#     id = []
#     tclass = []
#
#     for i in range(0,34):
#         name = str(i) + '_p4_cnn_result_data.csv'
#         x1 = pd.read_csv(path+name)
#         for i in range(len(x1)):
#             id.append(x1['id'][i])
#             tclass.append(int(x1['class'][i]))
#
#     english_column = pd.Series(id, name='id')
#     number_column = pd.Series(tclass, name='class')
#     predictions = pd.concat([english_column, number_column], axis=1)
#     predictions.to_csv(path + 'p4_cnn_result_data.csv', index=0, sep=',', columns=['id', 'class'])
#
# write_result1()

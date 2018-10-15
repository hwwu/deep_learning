#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/8/22 下午1:37
# @Author   :hwwu
# @File     :fasttext.py

import pandas as pd, numpy as np
import fastText

path = '/Users/liyangyang/Downloads/dwb/new_data/'
# column = "word_seg"
column = "article"


def write_train_data():
    train = pd.read_csv(path + 'train_set.csv')
    f = open(path + 't_train_set.txt', 'a')
    for i in range(80000):
        row = str(train[column][i]) + '\t' + '__myprefix__' + str(train['class'][i])
        f.write(row + '\n')
    f.close()
    f1 = open(path + 't_test_set.txt', 'a')
    for i in range(80000, len(train)):
        row = str(train[column][i]) + '\t' + '__myprefix__' + str(train['class'][i])
        f1.write(row + '\n')
    f1.close()
    # f2 = open(path + 'test_set1.txt', 'a')
    # for i in range(100000, len(train)):
    #     row = str(train[column][i])
    #     f2.write(row + '\n')
    # f2.close()


# write_train_data()

model_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/fasttext/'


def model():
    # model = fastText.train_supervised(path + 'train_set.txt', label='__myprefix__',bucket=400000
    #                                        ,wordNgrams=2,minCount=3,lr=1,lrUpdateRate=0)
    model = fastText.train_supervised(path + 't_train_set.txt', label='__myprefix__', bucket=39759
                                      , wordNgrams=3, minCount=3, lr=1, lrUpdateRate=200
                                      ,dim=128)
    result = model.test(path + 't_test_set.txt')
    print(result)
    # model.save_model(model_path + 'model')

    true_labels = []
    all_words = []
    f = open(path + 't_test_set.txt', 'r')
    for line in f:
        words, labels = model.get_line(line.strip())
        if len(labels) == 0:
            continue
        all_words.append(" ".join(words))
        true_labels += [labels]
    predictions, _ = model.predict(all_words)

    n = 0
    for i in range(len(true_labels)):
        if (predictions[i]==true_labels[i]):
            n+=1
    print(n/len(true_labels))

    # model = fastText.load_model(model_path + 'model')
    # id, all_words = get_test_words(model)
    # print('start predict data')
    # predictions, _ = model.predict(all_words)
    # print('predict data done')
    # write_result(id, predictions)

model()

def get_test_words(model):
    all_words = []
    id = []
    print('start read test set data')
    test = pd.read_csv(path + 'test_set.csv')
    for i in range(len(test)):
        words, _ = model.get_line(test[column][i].strip())
        all_words.append(" ".join(words))
        id.append(test['id'][i])
    print('read test set data done')
    return id, all_words


def write_result(id, predictions):
    r_id = []
    r_predictions = []
    for i in range(len(id)):
        r_id.append(int(id[i]))
        # price.append(round(x1[i][1],1))
        r_predictions.append(int(tostr(predictions[i])))

    english_column = pd.Series(r_id, name='id')
    number_column = pd.Series(r_predictions, name='class')
    predictions = pd.concat([english_column, number_column], axis=1)
    # another way to handle
    # save = pd.DataFrame({'user_id': user_id, 'prediction_pay_price': price})
    predictions.to_csv(path + 'result_data.csv', index=0, sep=',', columns=['id', 'class'])


def tostr(s):
    s = str(s).replace('__myprefix__', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('\'', '')
    return s

# model()

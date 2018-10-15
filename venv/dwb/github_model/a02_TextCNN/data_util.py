# -*- coding: utf-8 -*-
import codecs
import random
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import os
import pickle

local_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/github_model/a02_TextCNN/'


def load_data_multilabel(traning_data_path,vocab_word2index, vocab_label2index,sentence_len,training_portion=0.9):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    random.shuffle(lines)
    label_size=len(vocab_label2index)
    X = []
    Y = []
    for i,line in enumerate(lines):
        raw_list = line.strip().split("__myprefix__")
        input_list = raw_list[0].strip().split(" ")
        input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
        x=[vocab_word2index.get(x,0) for x in input_list]
        label_list = raw_list[1:]
        label_list=[l.strip().replace(" ", "") for l in label_list if l != '']
        label_list=[vocab_label2index[label] for label in label_list]
        # y=transform_multilabel_as_multihot(label_list,label_size)
        y=label_list
        X.append(x)
        Y.append(y)
    Y = np.array(Y).reshape(-1)
    X = pad_sequences(X, maxlen=sentence_len, value=0.)  # padding to max length
    number_examples = len(lines)
    training_number=int(training_portion* number_examples)
    train = (X[0:training_number], Y[0:training_number])
    valid_number=number_examples-training_number
    test = (X[training_number+ 1:training_number+valid_number+1], Y[training_number + 1:training_number+valid_number+1])
    return train,test


def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(training_data_path,vocab_size,name_scope='cnn'):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """

    cache_vocabulary_label_pik=local_path+'cache'+"_"+name_scope # path to save cache
    if not os.path.isdir(cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    # if cache exists. load it; otherwise create it.
    cache_path =cache_vocabulary_label_pik+"/"+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            return pickle.load(data_f)
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        # vocabulary_word2index[_PAD]=PAD_ID
        # vocabulary_index2word[PAD_ID]=_PAD
        # vocabulary_word2index[_UNK]=UNK_ID
        # vocabulary_index2word[UNK_ID]=_UNK

        vocabulary_label2index={}
        vocabulary_index2label={}

        #1.load raw data
        file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
        lines=file_object.readlines()
        #2.loop each line,put to counter
        c_inputs=Counter()
        c_labels=Counter()
        for line in lines:
            raw_list=line.strip().split("__myprefix__")

            input_list = raw_list[0].strip().split(" ")
            input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
            label_list=[l.strip().replace(" ","") for l in raw_list[1:] if l!='']
            c_inputs.update(input_list)
            c_labels.update(label_list)
        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        label_list=c_labels.most_common()
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_=tuplee
            vocabulary_word2index[word]=i+1
            vocabulary_index2word[i+1]=word

        for i,tuplee in enumerate(label_list):
            label,_=tuplee;label=str(label)
            vocabulary_label2index[label]=i
            vocabulary_index2label[i]=label

        #save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label

def get_target_label_short(eval_y):
    eval_y_short=[] #will be like:[22,642,1391]
    for index,label in enumerate(eval_y):
        if label>0:
            eval_y_short.append(index)
    return eval_y_short

# training_data_path = '/Users/liyangyang/Downloads/bdci/train.txt'
# vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label= \
#     create_vocabulary(training_data_path,17259,name_scope='cnn')
# vocab_size = len(vocabulary_word2index);print("cnn_model.vocab_size:",vocab_size);num_classes=len(vocabulary_index2label);print("num_classes:",num_classes)
# print(vocabulary_index2label)
# train, test= load_data_multilabel(training_data_path,vocabulary_word2index, vocabulary_label2index,200)
# trainX, trainY = train
# testX, testY = test
# #print some message for debug purpose
# print("length of training data:",len(trainX),";length of validation data:",len(testX))
# print("trainX[0]:", trainX[1]);
# print("trainY[0]:", trainY[1])
# # train_y_short = get_target_label_short(trainY[1])
# # print("train_y_short:", train_y_short)
# for i in range(1,100):
#     print(vocabulary_index2word[i])
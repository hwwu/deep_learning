#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/9/12 下午6:39
# @Author   :hwwu
# @File     :snownlp1.py
import pickle
import numpy as np


def readdumpobj(path):
    file = open(path, "rb")
    bunch = pickle.load(file)
    file.close()
    return bunch


def outemotionword(path):
    emotionset = []
    with open(path, "rb") as fp:
        for word in fp:
            if not word.isspace():
                word = word.decode("utf-8")
                emotionset.append(word.strip())
    return emotionset


def loadDataSet(path):  # path是为了读入将情感词典
    postingList = readdumpobj("D:\linguistic-corpus\postingList\postingList.dat")
    classVec = readdumpobj("D:\linguistic-corpus\postingList\classVec.dat")
    emotionset = outemotionword(path)
    return postingList, classVec, emotionset


class NBayes(object):
    def __init__(self):
        self.vocabulary = []  # 词典，文本set表
        self.idf = 0  # 词典的idf权值向量
        self.tf = 0  # 训练集的权值矩阵
        self.tdm = 0  # P(x|yi)
        self.Pcates = {}  # P(yi)--是个类别字典
        self.labels = []  # 对应每个文本的分类，是个外部导入的列表[0,1,0,1,0,1]
        self.doclength = 0  # 训练集文本数，训练文本长度
        self.vocablen = 0  # 词典词长,self.vocabulary长度
        self.testset = 0  # 测试集

    #   加载训练集并生成词典，以及tf, idf值
    def train_set(self, trainset, classVec, emotionset):
        self.cate_prob(classVec)  # 计算每个分类在数据集中的概率：P(yi)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for word in emotionset]  # 生成词典
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        # self.calc_wordfreq(trainset)
        self.calc_tfidf(trainset)  # 生成tf-idf权值
        self.build_tdm()  # 按分类累计向量空间的每维值：P(x|yi)

    # 生成 tf-idf
    def calc_tfidf(self, trainset):
        self.idf = np.zeros([1, self.vocablen])
        self.tf = np.zeros([self.doclength, self.vocablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                if word in self.vocabulary:
                    self.tf[indx, self.vocabulary.index(word)] += 1
            # 消除不同句长导致的偏差
            self.tf[indx] = self.tf[indx] / float(len(trainset[indx]))
            for signleword in set(trainset[indx]):
                if signleword in self.vocabulary:
                    self.idf[0, self.vocabulary.index(signleword)] += 1
        self.idf = np.log(float(self.doclength) / (self.idf + 1))  # 防止该词语不在语料中，就会导致分母为零
        self.tf = np.multiply(self.tf, self.idf)  # 矩阵与向量的点乘

    # 生成普通的词频向量
    def calc_wordfreq(self, trainset):
        self.idf = np.zeros([1, self.vocablen])  # 1*词典数
        self.tf = np.zeros([self.doclength, self.vocablen])  # 训练集文件数*词典数
        for indx in range(self.doclength):  # 遍历所有的文本
            for word in trainset[indx]:  # 遍历文本中的每个词
                if word in self.vocabulary:
                    self.tf[indx, self.vocabulary.index(word)] += 1  # 找到文本的词在字典中的位置+1
            for signleword in set(trainset[indx]):
                if signleword in self.vocabulary:
                    self.idf[0, self.vocabulary.index(signleword)] += 1

    # 计算每个分类在数据集中的概率：P(yi)
    def cate_prob(self, classVec):
        self.labels = classVec
        labeltemps = set(self.labels)  # 获取全部分类
        for labeltemp in labeltemps:
            # 统计列表中重复的值：self.labels.count(labeltemp)
            self.Pcates[labeltemp] = float(self.labels.count(labeltemp)) / float(len(self.labels))

    # 按分类累计向量空间的每维值：P(x|yi)
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates), self.vocablen])  # 类别行*词典列
        sumlist = np.zeros([len(self.Pcates), 1])  # 统计每个分类的总值
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx]  # 将同一类别的词向量空间值加总
            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])  # 统计每个分类的总值--是个标量
        self.tdm = self.tdm / sumlist  # P(x|yi)

    # 测试集映射到当前词典
    def map2vocab(self, testdata):
        self.testset = np.zeros([1, self.vocablen])
        # 删除测试集中词不在训练集中
        for word in testdata:
            if word in self.vocabulary:
                self.testset[0, self.vocabulary.index(word)] += 1

    # 输出分类类别
    def predict(self, testset):
        if np.shape(testset)[1] != self.vocablen:
            print("输入错误")
            exit(0)
        predvalue = 0
        predclass = ""
        for tdm_vect, keyclass in zip(self.tdm, self.Pcates):
            # P(x|yi)P(yi)
            temp = np.sum(testset * tdm_vect * self.Pcates[keyclass])
            if temp > predvalue:
                predvalue = temp
                predclass = keyclass
        return predclass


if __name__ == "__main__":
    postingList, classVec, emotionset = loadDataSet("D:\sentiment-word\emotionword.txt")
    testset = postingList[119]
    nb = NBayes()  # 类的实例化
    nb.train_set(postingList, classVec, emotionset)  # 训练数据集
    nb.map2vocab(testset)  # 随机选择一个测试句，这里2表示文本中的第三句话，不是脏话，应输出0。
    print(nb.predict(nb.testset))  # 输出分类结果0表示消极，1表示积极
    print("分类结束")

# !/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/6/21 下午1:27
# @Author   :hwwu
# @File     :PricePredictor.py

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

# def character(id,train_x):
#     r =[]
#     r.append(id)
#     r.append(train_x[10]*0.6+train_x[11]*0.15+train_x[12]*0.25)
#     r.append(train_x[13]*0.6+train_x[14]*0.15+train_x[15]*0.25)
#     r.append(train_x[16]**(1/2))
#     for i in [0,1,2,4,5,6,10,11,12,13,14,15,17,18]:
#         r.append(train_x[i])
#         for j in range(i,19):
#             r.append(train_x[i]+train_x[j])
#             r.append(train_x[i]-train_x[j])
#             r.append(train_x[i]*train_x[j])
#             r.append(train_x[i]/(train_x[j]+0.1))
#
#     return r


from sklearn import preprocessing


def load_train_data():
    min_max_scaler = preprocessing.MinMaxScaler()
    train_ = read_data.read_result_data('public.train.csv')
    train_x = train_[:, 2:21]
    train_y = train_[:, 21]
    train_x = min_max_scaler.fit_transform(train_x)

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
                # x.append(character(id,train_x[i]))
                x.append([id,
                          train_x[i][0],
                          train_x[i][1],
                          train_x[i][3],
                          train_x[i][2] * train_x[i][4],
                          train_x[i][2] * train_x[i][5],
                          train_x[i][2] * train_x[i][6],
                          train_x[i][13],
                          train_x[i][14],
                          train_x[i][15],
                          train_x[i][16],
                          train_x[i][17],
                          train_x[i][18]
                          ])
                y.append(abs(train_y[i]))
    print(len(x))
    # for i in range(10):
    #     print(x[i])
    return x, y


def load_test_data(file='public.test.csv'):
    # train_ = read_data.read_result_data('test_data_all.csv')
    train_ = read_data.read_result_data(file)
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
                # x.append(character(id, train_x[i]))
                y.append(abs(train_y[i]))
    for i in range(1):
        print(x[i])
    print(len(x))
    return x, y


# 对训练集和测试集分别进行交叉验证，得到error measure for official scoring : RMSE

x, y = load_train_data()

X_train = x[0:8000:1]
y_train = y[0:8000:1]
X_test = x[8000:8905:1]
y_test = y[8000:8905:1]

#
# x1, y1 = load_test_data()
# X_test = x1[0::1]
# y_test = y1[0::1]

# x2, y2 = load_test_data(file='test_data_all.csv')
# X_test_1 = x2[0::1]
# y_test_1 = y2[0::1]
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import xgboost

n_folds = 5


def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


def rmse_cv_test(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((len(X), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1, max_iter=100000))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3, max_iter=100000))
KRR = KernelRidge(alpha=0.6, kernel='linear', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=30000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.005, max_depth=23,
                             max_delta_step=100000,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(num_leaves=31,
                              learning_rate=0.03,
                              n_estimators=1000,
                              subsample=.9,
                              colsample_bytree=.9,
                              random_state=666)

averaged_models = AveragingModels(models=(lasso, ENet, KRR, model_lgb, model_xgb))

stacked_averaged_models = StackingAveragedModels(base_models=(lasso, ENet, model_lgb),
                                                 meta_model=model_xgb)

#
# score = rmse_cv(averaged_models)
# score_test = rmse_cv_test(averaged_models)
averaged_models.fit(X_train, y_train)
y_ = averaged_models.predict(X_test)


# y_1 = model_xgb.predict(X_test_1)
# y_1= averaged_models.predict(X_test_1)


# stacked_averaged_models.fit(X_train, y_train)
# y_1 = stacked_averaged_models.predict(X_test)

# stacked_averaged_models.fit(X_train, y_train)
# y_2 = stacked_averaged_models.predict(X_test)
# # y_1= stacked_averaged_models.predict(X_test_1)
#
# y_ = []
# for i in range(len(y_2)):
#     y_.append([y_2[i] * 0 + y_3[i] * 1])

# r = []
# for i in range(8338):
#     id = y_test[i][0]
#     p = y_[i]
#     r.append([id, p])
# np.savetxt('/Users/liyangyang/Downloads/datafountain/guangdianfute/test_data_3', r)
#
#
# r1 = []
# for i in range(17243):
#     id = y_test_1[i][0]
#     p = y_1[i]
#     r1.append([id, p])
# np.savetxt('/Users/liyangyang/Downloads/datafountain/guangdianfute/test_data_all_1', r1)

# for i in range(10):
#     print(y_test[i], y_[i])
#
#
def rmse_my(y_test, y_, s):
    error = []
    n = 0
    for i in range(len(y_test)):
        if ((y_test[i] - y_[i]) * (y_test[i] - y_[i]) > 1):
            # print(X_test[i], y_test[i], y_[i])
            n += 1
        error.append(y_test[i] - y_[i])
    print('n', n)

    squaredError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方

    print(s, "Square Error: ", sorted(squaredError, reverse=True))

    print(s, "MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
    from math import sqrt

    print(s, "RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE


#
rmse_my(y_test, y_, 'y_')
# rmse_my(y_train,y_t,'y_t')

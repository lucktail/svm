# -*- coding: utf-8 -*-

import numpy as np
from sklearn import model_selection as ms
from sklearn import svm

print 'helloWorld'


# def iris_type(s):
#     it = {'\"versicolor\"': 0, '\"virginica\"': 1, '\"setosa\"': 2}
#     return it[s]

i = -1


def preCalGongye(s):
    global i
    if i == -1:
        i = float(s)
        return 0
    else:
        if float(s) >= i:
            i = float(s)
            return 1
        else:
            i = float(s)
            return 0


def preCal300(s):
    if float(s) > 0:
        return 1
    else:
        return 0

# path = './data/iris.txt'
# data = np.loadtxt(path, dtype=float, usecols=(1, 2, 3, 4, 5), delimiter=' ', converters={5: iris_type})


path = './data/srisk.txt'
# 总srisk vs 工业增加值增长率
data = np.loadtxt(path, dtype=float, usecols=(1, 6, 7, 8, 9, 4), delimiter=',', converters={4: preCalGongye})
# 总srisk vs 沪深300指数收益率
# data = np.loadtxt(path, dtype=float, usecols=(1, 7), delimiter=',', converters={7: preCal300})
print data

# x, y = np.split(data, (4,), axis=1)
# x = x[:, :2]
# x_train, x_test, y_train, y_test = ms.train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)

# cut 1--130
cut = 80
x, y = np.split(data, (5,), axis=1)
x_train = x[1:cut]
y_train = y[1:cut]
x_test = x[cut:]
y_test = y[cut:]
print x_train


# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

print clf.score(x_train, y_train)
# y_hat = clf.predict(x_train)
# show_accuracy(y_hat, y_train, 'train')

print clf.score(x_test, y_test)

# predicted = clf.predict([[5.1, 3.5], [1.1, 2.2]])
# print predicted

# # "1" 5.1 3.5 1.4 0.2 "setosa"
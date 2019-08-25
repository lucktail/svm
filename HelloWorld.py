import numpy as np
#import sklearn as sk
from sklearn import model_selection as ms
from sklearn import svm

print 'helloWorld'
def iris_type(s):
    it = {'\"versicolor\"': 0, '\"virginica\"': 1, '\"setosa\"': 2}
    return it[s]
path = 'C:/Users/lucktail_lun/Desktop/data/iris.txt'
data = np.loadtxt(path, dtype=float, usecols=(1, 2, 3, 4, 5), delimiter=' ', converters={5: iris_type})

x, y = np.split(data, (4,), axis=1)
x = x[:, :2]
x_train, x_test, y_train, y_test = ms.train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
#clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

print clf.score(x_train, y_train)
#y_hat = clf.predict(x_train)
#show_accuracy(y_hat, y_train, 'train')

print clf.score(x_test, y_test)

predited = clf.predict([[5.1, 3.5],[1.1, 2.2]])
print predited

#"1" 5.1 3.5 1.4 0.2 "setosa"
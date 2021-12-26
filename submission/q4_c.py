"""
from sklearn import linear_model
import numpy
from sklearn.metrics import accuracy_score
from numpy import genfromtxt

train_data = genfromtxt('train.csv', delimiter=',')
numpy.random.shuffle(train_data)
num_values = train_data.shape[1]
n_samples = train_data.shape[0]
X = numpy.array(train_data[:,0:num_values-1])
Y = numpy.array(train_data[:,num_values-1])

k = int(numpy.floor(n_samples * 20.0 / 100.0))
X_validate = X[:k, :]
Y_validate = Y[:k]
X_train = X[k:, :]
Y_train = Y[k:]

alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
l1_ratios = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
#alphas = [0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]

for alpha in alphas:
    for l1_ratio in l1_ratios:
        clf = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        clf.fit(X_train, Y_train)

        predictions = clf.predict(X_validate)

        thres = 0.50
        predictions[predictions > thres] = 1
        predictions[predictions <= thres] = 0

        accuracy = accuracy_score(Y_validate, predictions)
        print alpha, ",", l1_ratio, " -> ", accuracy
"""

from sklearn import linear_model
import numpy
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
import sys
import os

if len(sys.argv) != 3:
    print "Invalid command line argument"
    exit(1)

train_file = sys.argv[1]
test_file = sys.argv[2]

train_data = genfromtxt(train_file, delimiter=',')
test_data = genfromtxt(test_file, delimiter=',')

num_values = train_data.shape[1]
X_train = numpy.array(train_data[:,0:num_values-1])
Y_train = numpy.array(train_data[:,num_values-1])

"""
X_test = numpy.array(test_data[:,0:num_values-1])
Y_test = numpy.array(test_data[:,num_values-1])
"""
X_test = numpy.array(test_data)

alpha = 0.0001
l1_ratio = 0.1
clf = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)

thres = 0.50
predictions[predictions > thres] = 1
predictions[predictions <= thres] = 0

for i in predictions:
    print int(i)

"""
accuracy = accuracy_score(Y_test, predictions)
print alpha, accuracy
"""

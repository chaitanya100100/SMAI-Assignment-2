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

clf = linear_model.LinearRegression()
clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)

thres = 0.50
predictions[predictions > thres] = 1
predictions[predictions <= thres] = 0

for i in predictions:
    print int(i)

"""
accuracy = accuracy_score(Y_test, predictions)
print accuracy
"""


from numpy import genfromtxt
import numpy as np
from random import randint
import PIL.Image
from cStringIO import StringIO
import IPython.display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing


X_train = genfromtxt('../q3/notMNIST_train_data.csv', delimiter=',')
Y_train = genfromtxt('../q3/notMNIST_train_labels.csv', delimiter=',')
X_test = genfromtxt('../q3/notMNIST_test_data.csv', delimiter=',')
Y_test = genfromtxt('../q3/notMNIST_test_labels.csv', delimiter=',')



def showarray(a, nam, fmt='png'):
    a = np.uint8(a)
    #k = PIL.Image.fromarray(a).save(f, fmt)
    k = PIL.Image.fromarray(a)
    #k.save(nam)
    print nam
    k.show()
    #raw_input()
    #IPython.display.display(IPython.display.Image(data=f.getvalue()))


"""
clf = LogisticRegression(penalty='l1', C=0.001)
clf.fit(X_train, Y_train)
Y_test_predicted = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_test_predicted)

#print penalty, " , ", C, "  : ", acc
W = clf.coef_[0]
W1 = W


clf = LogisticRegression(penalty='l2', C=0.001)
clf.fit(X_train, Y_train)
Y_test_predicted = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_test_predicted)

# print penalty, " , ", C, "  : ", acc
W = clf.coef_[0]
W2 = W

#W = abs(W)
#min_val = np.amin(W)
#W = np.log(W - min_val + 1)
min_val = min(np.amin(W1), np.amin(W2))
max_val = max(np.amax(W1), np.amax(W2))

#W1 = (W1 - min_val) / (max_val - min_val)
#W2 = (W2 - min_val) / (max_val - min_val)

W1 = (W1 - min_val) + 1
W2 = (W2 - min_val) + 1

W1 = np.log(W1)
W2 = np.log(W2)

min_val = min(np.amin(W1), np.amin(W2))
max_val = max(np.amax(W1), np.amax(W2))
W1 = (W1 - min_val) / (max_val - min_val)
W2 = (W2 - min_val) / (max_val - min_val)

W1 = W1 * 253 + 1
W2 = W2 * 253 + 1

showarray(W1.reshape(28, 28), "l1" + "_" + str(0.001) + ".png")
showarray(W2.reshape(28, 28), "l2" + "_" + str(0.001) + ".png")
#print W

exit(0)
"""


penalties = ['l1']
Cs = [0.003]
di = "."

"""
penalties = ['l1']
Cs = [1.0]
"""
for penalty in penalties:
    for C in Cs:
        clf = LogisticRegression(penalty=penalty, C=C)
        clf.fit(X_train, Y_train)
        Y_test_predicted = clf.predict(X_test)
        acc = accuracy_score(Y_test, Y_test_predicted)

        print penalty, " , ", C, "  : ", acc
        W = clf.coef_[0]


        #W = abs(W)
        #min_val = np.amin(W)
        #W = np.log(W - min_val + 1)
        min_val = np.amin(W)
        max_val = np.amax(W)
        W = (W - min_val) / (max_val - min_val)
        #W = np.log(W + 1)
        W = W * 253 + 1
        showarray(W.reshape(28, 28), di + "/" + penalty + "_" + str(C) + ".png")
        #print W

"""
def showarray(a, fmt='png'):
    a = np.uint8(a)
    #k = PIL.Image.fromarray(a).save(f, fmt)
    k = PIL.Image.fromarray(a)
    k.show()
    #IPython.display.display(IPython.display.Image(data=f.getvalue()))


n = randint(1, 1000)
label = Y_test[n]
im = X_test[n, :].reshape((28, 28))
print(label)
#showarray(im)
"""

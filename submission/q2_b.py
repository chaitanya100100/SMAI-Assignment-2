import sys
import os
import numpy as np
import cPickle
import re

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
K.set_image_dim_ordering('tf')

from sklearn import svm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


np.random.seed(123)

def display_im(im):
    pass

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = cPickle.load(fo)
    return dict_

def unpack_batch(file):
    dict_ = unpickle(file)
    X = dict_["data"]
    X = X.reshape(X.shape[0], 3, 32, 32)
    X = np.rollaxis(X, 1, 4)
    X = X.astype('float32')
    X = X / 255.0
    Y = np.asarray(dict_["labels"])
    Y = np_utils.to_categorical(Y, num_classes)
    return X, Y


# command line arguments
if len(sys.argv) != 3:
    print "Error in command line arguments"
    exit(1)
train_dir = sys.argv[1]
test_file = sys.argv[2]

dict_ = unpickle(os.path.join(train_dir, 'batches.meta'))
label_names = dict_["label_names"]
#print label_names
num_classes = len(label_names)
train_batches = [f for f in os.listdir(train_dir) if re.match(r'data_batch_[0-9]+', f)]


# make training data
X_train = []
Y_train = []
for batch in train_batches:
    X, Y = unpack_batch(os.path.join(train_dir, batch))
    X_train.append(X)
    Y_train.append(Y)

X_train = np.concatenate(X_train)
Y_train = np.concatenate(Y_train)

n_sample = X_train.shape[0]
k = int(0.001 * n_sample)
X_validate = X_train[:k]
Y_validate = Y_train[:k]
X_train = X_train[k:]
Y_train = Y_train[k:]


#------------------------------------------------
# load test file
dict_ = unpickle(test_file)
X_test = dict_["data"]
X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
X_test = np.rollaxis(X_test, 1, 4)
X_test = X_test.astype('float32')
X_test = X_test / 255.0
#X_test, Y_test = unpack_batch(test_file)

"""
print X_train.shape
print Y_train.shape
print X_validate.shape
print Y_validate.shape
print X_test.shape
#print Y_test.shape
"""
"""
X_train = X_train[:10]
Y_train = Y_train[:10]
X_validate = X_validate[:10]
Y_validate = Y_validate[:10]
X_test = X_test[:10]
#Y_test = Y_test[:10]
"""

# create model
model = Sequential()
model.add(Conv2D(
    16,
    (3, 3),
    input_shape=(32, 32, 3),
    padding='same',
    #activation='relu'
))
model.add(Conv2D(
    16,
    (3, 3),
    input_shape=(32, 32, 3),
    padding='same',
    #activation='relu'
))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(
    32,
    (3, 3),
    #activation='relu',
    padding='same'
))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

fc1 = Dense(256, activation='relu')
model.add(fc1)

model.add(Dropout(0.2))
fc2 = Dense(128, activation='relu')
model.add(fc2)
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# compile
epochs = 40
lrate = 0.01
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#print(model.summary())

model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=epochs, batch_size=32)
prediction_vectors = model.predict(X_test)
predictions = np.argmax(prediction_vectors, axis=1)
#print prediction_vectors
#print predictions

f = open('q2_b_output.txt', 'w')
for p in predictions:
    print >> f, label_names[p]
f.close()

"""
scores = model.evaluate(X_test, Y_test, verbose=1)
print
print "Accuracy: %.2f%%" % (scores[1]*100)
"""

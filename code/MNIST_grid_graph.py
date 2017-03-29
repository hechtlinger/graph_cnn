"""
MNIST expriment exploring the performance of the graph convolution
when the graph structure connect a pixel to its 8 adjunct neighbors. 

This reproduce the results described in section 4.2.
"""

### Dependencies 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils

from graph_convolution import GraphConv
from grid_mat_utils import generate_Q

import numpy as np

np.random.seed(1984) # for reproducibility

### Parameters
batch_size=128
nb_epoch=150
nb_neighbors = 25
nb_filter = nb_layer_1 = 50
nb_layer_2 = 100
nb_classes = 10
pool_size = (2, 2)
results = dict()

### Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# %% Generate the transition matrix Q
q_mat_layer1 = generate_Q(28,3)
q_mat_layer2 =  generate_Q(14,3)

q_mat_layer1 = np.argsort(q_mat_layer1,1)[:,-nb_neighbors:]
q_mat_layer2 = np.argsort(q_mat_layer2,1)[:,-nb_neighbors:]

# %% Standard LeNet structure with Graph Convolution
model = Sequential()
model.add(GraphConv(nb_filter=nb_layer_1, Q_matrix = q_mat_layer1, 
                    nb_neighbors=nb_neighbors, activation='relu',
                    input_shape=(X_train.shape[1],1,)))
model.add(Reshape((28, 28, nb_layer_1)))
model.add(MaxPooling2D(pool_size=pool_size, dim_ordering='tf'))
model.add(Dropout(0.25))
model.add(Reshape((196, nb_layer_1)))
model.add(GraphConv(nb_filter=nb_layer_2, Q_matrix = q_mat_layer2, 
                    nb_neighbors=nb_neighbors, activation='relu', bias = True))
model.add(Reshape((14, 14, nb_layer_2)))
model.add(MaxPooling2D(pool_size=pool_size, dim_ordering='tf'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', 
              metrics=['accuracy'])

results['two_layer_cnn'] = model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          nb_epoch=nb_epoch,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))
                          
graph_lenet_error=1-results['two_layer_cnn'].__dict__['history']['val_acc'][-1]
        
print('Error rate for Graph LeNet: %.04f'%graph_lenet_error)

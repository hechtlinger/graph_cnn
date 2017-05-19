'''
MNIST experiment comparing the performance of the graph convolution
with Logistic regression and fully connected neural networks. 

The data graph structured is learned from the correlation matrix. 

This reproduce the results shown in table 2 of:
"A generalization of Convolutional Neural Networks to Graph-Structured Data"
'''

### Dependencies 
import cPickle as pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import to_categorical

from graph_convolution import GraphConv

import numpy as np

from sklearn.preprocessing import normalize

np.random.seed(2017) # for reproducibility

### Parameters
batch_size=128
epochs=40
num_neighbors=6
filters = 20
num_classes = 10
results = dict()

### Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

ix = np.where(np.var(X_train,axis=0)>0)[0] 
X_train = X_train[:,ix]
X_test = X_test[:,ix]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

### Prepare the Graph Correlation matrix 
corr_mat = np.array(normalize(np.abs(np.corrcoef(X_train.transpose())), 
                              norm='l1', axis=1),dtype='float64')
graph_mat = np.argsort(corr_mat,1)[:,-num_neighbors:]

# %%

### Single layer of Graph Convolution
g_model = Sequential()
g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat, 
                      num_neighbors=num_neighbors, activation='relu',
                      input_shape=(X_train.shape[1],1,)))
g_model.add(Dropout(0.2))
g_model.add(Flatten())
g_model.add(Dense(10, activation='softmax'))

g_model.summary()

g_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), 
                metrics=['accuracy'])

results['g'] = g_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))

g_error = 1-results['g'].__dict__['history']['val_acc'][-1]
# %%

### Graph Convolution followed by FC layer
g_fc_model = Sequential()
g_fc_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat, 
                         num_neighbors=num_neighbors, activation='relu', 
                         input_shape=(X_train.shape[1],1,)))
g_fc_model.add(Dropout(0.2))
g_fc_model.add(Flatten())
g_fc_model.add(Dense(512, activation='relu'))
g_fc_model.add(Dropout(0.2))
g_fc_model.add(Dense(10, activation='softmax'))

g_fc_model.summary()

g_fc_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), 
                   metrics=['accuracy'])

results['g_fc'] = g_fc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))

g_fc_error = 1-results['g_fc'].__dict__['history']['val_acc'][-1]
# %%

### 2 Layer of Graph Convolution
g_g_model = Sequential()
g_g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat, 
                        num_neighbors=num_neighbors, activation='relu', 
                        bias = True, input_shape=(X_train.shape[1],1,)))
g_g_model.add(Dropout(0.2))
g_g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat, 
                        num_neighbors=num_neighbors, activation='relu'))
g_g_model.add(Dropout(0.2))
g_g_model.add(Flatten())
g_g_model.add(Dense(10, activation='softmax'))

g_g_model.summary()

g_g_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), 
                  metrics=['accuracy'])

results['g_g'] = g_g_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))

g_g_error = 1-results['g_g'].__dict__['history']['val_acc'][-1]
# %%

### Fully Connected - FC Model
FC_FC_model = Sequential()
FC_FC_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
FC_FC_model.add(Dropout(0.2))
FC_FC_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
FC_FC_model.add(Dropout(0.2))
FC_FC_model.add(Dense(10))
FC_FC_model.add(Activation('softmax'))

FC_FC_model.summary()

FC_FC_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), 
                    metrics=['accuracy'])

results['fc_fc'] = FC_FC_model.fit(X_train, Y_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  verbose = 2,
                                  validation_data=(X_test, Y_test))

fc_fc_error = 1-results['fc_fc'].__dict__['history']['val_acc'][-1]
# %%

### Logistic Regression Model
LR_model = Sequential()
LR_model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
LR_model.add(Activation('softmax'))

LR_model.summary()

LR_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), 
                 metrics=['accuracy'])

results['lr'] = LR_model.fit(X_train, Y_train,
                                epochs=300,
                                batch_size=batch_size,
                                verbose = 2,
                                validation_data=(X_test, Y_test))

lr_error = 1-results['lr'].__dict__['history']['val_acc'][-1]

# %%
print('Error rates for the different models:')
print('Logistic Regression: %.04f'%lr_error)
print('1 Layer of graph convolution: %.04f'%g_error)
print('2 Layers of graph convolution: %.04f'%g_g_error)
print('1 Layers f graph convolution & 1 FC layer: %.04f'%g_fc_error)
print('2 FC layers: %.04f'%fc_fc_error)

#pickle.dump(results, open('results/MNIST_results.p','wb'))

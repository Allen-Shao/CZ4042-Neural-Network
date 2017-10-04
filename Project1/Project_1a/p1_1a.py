
# coding: utf-8

# In[5]:

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers, callbacks
import matplotlib.pyplot as plt


# In[6]:

import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.utils import np_utils
import os
import math


# In[7]:

plot_pic_folder = "./pic/"
#  create folder if not exists
if not os.path.exists(plot_pic_folder):
    os.makedirs(plot_pic_folder)
num_of_question = 5
for i in range(1,num_of_question+1):
    if not os.path.exists(plot_pic_folder + str(i) + '/'):
        os.makedirs(plot_pic_folder + str(i) + '/')


# In[ ]:




# In[8]:

class BatchTime(callbacks.Callback):
    def __init__(self):
        self.logs=[]
        self.batch_time = []
        self.start_time = []
        self.cur_start_time = 0
        
        
    def on_batch_begin(self, batch, logs={}):
        self.cur_start_time = time.time()
        self.start_time.append(self.cur_start_time)
        return
 
    def on_batch_end(self, batch, logs={}):
        self.batch_time.append(time.time() - self.cur_start_time)
        return


# In[9]:

# scale data
def scale(X, X_min, X_max):
    # min-max normalization
    return (X - X_min)/(X_max-np.min(X, axis=0))


# In[10]:

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


# In[20]:

# Point out the maximum and minimum point

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "MAX Point x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

def annot_min(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "MIN Point x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="bottom")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.96), **kw)


# In[12]:

#read train data
train_input = np.loadtxt('./data/sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
trainX = scale(trainX, trainX_min, trainX_max)

train_Y[train_Y == 7] = 6
trainY = np.zeros((train_Y.shape[0], 6))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1


#read test data
test_input = np.loadtxt('./data/sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

# testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
testX = scale(testX, trainX_min, trainX_max)

test_Y[test_Y == 7] = 6
testY = np.zeros((test_Y.shape[0], 6))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1

del test_Y, train_Y, trainX_min, trainX_max
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)


# In[22]:

# question 1 and question 2:
decay = 1e-6
learning_rate = 0.01
epoch_num = 5

batch_time_taken = []
total_time_taken = []
batch_sizes = [4,8,16,32,64]
for batch_size in batch_sizes:
    bt = BatchTime()
    inputs = Input(shape=(trainX.shape[1],))
    dense1 = Dense(10, activation='relu')(inputs)
    predictions = Dense(trainY.shape[1], activation='softmax')(dense1)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=learning_rate, decay=decay)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    start_time = time.time()
    train_his = model.fit(trainX, trainY, epochs=epoch_num, verbose=2, 
                          validation_data=[testX,testY],
                          batch_size = batch_size,
                          callbacks=[bt])
    total_time_taken.append(time.time() - start_time)
    
#     loss, acc = model.evaluate(testX, testY, verbose=2)

    
    # Plot the training error and the test accuracy against number of epochs
    plt.figure()
    annot_min(range(1, 1+epoch_num), np.array(train_his.history['loss']))
    plt.plot(range(1, 1+epoch_num), train_his.history['loss'], label='train_cost')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.savefig(plot_pic_folder + '2/p1a_batchsize' + str(batch_size) + '_cost.png')
    plt.gcf().clear()
    plt.close()

    plt.figure()
    annot_max(range(1, 1+epoch_num), np.array(train_his.history['val_acc']))
    plt.plot(range(1, 1+epoch_num), train_his.history['val_acc'], label='test_accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')

    plt.savefig(plot_pic_folder + '2/p1a_batchsize' + str(batch_size) + '_sample_accuracy.png')
    plt.gcf().clear()
    plt.close()
    
    batch_time_taken.append(np.mean(bt.batch_time))

# Plot the time taken
batch_time_taken = np.array(batch_time_taken) * 1000
plt.figure()
plt.plot(batch_sizes, batch_time_taken, label='time_taken')
plt.xlabel('batch_sizes')
plt.xticks(batch_sizes)
plt.ylabel('time_taken(ms)')
plt.title('time for a weight update')
plt.savefig(plot_pic_folder + '2/p1a_batch_time.png')
# plt.show()
plt.gcf().clear()
plt.close()

total_time_taken = np.array(total_time_taken) * 1000
plt.figure()
plt.plot(batch_sizes, total_time_taken, label='time_taken')
plt.xlabel('batch_sizes')
plt.xticks(batch_sizes)
plt.ylabel('time_taken(ms)')
plt.title('total time for training')
plt.savefig(plot_pic_folder + '2/p1a_total_time.png')
# plt.show()
plt.gcf().clear()
plt.close()


# In[ ]:




# In[ ]:

#question 3

decay = 1e-6
learning_rate = 0.01
epoch_num = 2

batch_time_taken = []
total_time_taken = []
neuron_nums = [5,10,15,20,25]
for neuron_num in neuron_nums:
    bt = BatchTime()

    inputs = Input(shape=(trainX.shape[1],))
    dense1 = Dense(neuron_num, activation='sigmoid')(inputs)
    predictions = Dense(trainY.shape[1], activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=learning_rate, decay=decay)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    start_time = time.time()
    train_his = model.fit(trainX, trainY, epochs=epoch_num, verbose=2, 
                          validation_data=[testX,testY],
                          callbacks=[bt],
                          batch_size = 32)
    total_time_taken.append(time.time() - start_time)
#     loss, acc = model.evaluate(testX, testY, verbose=2)

    
    # Plot the training error and the test accuracy against number of epochs
    plt.figure()
    plt.plot(range(1, 1+epoch_num), train_his.history['loss'], label='train_cost')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    annot_min(range(1, 1+epoch_num), np.array(train_his.history['loss']))
    plt.savefig(plot_pic_folder + '3/p1a_neuron' + str(neuron_num) + '_cost.png')
    plt.gcf().clear()
    plt.close()
    

    plt.figure()
    plt.plot(range(1, 1+epoch_num), train_his.history['val_acc'], label='test_accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    annot_max(range(1, 1+epoch_num), np.array(train_his.history['val_acc']))
    plt.savefig(plot_pic_folder + '3/p1a_neuron' + str(neuron_num) + '_sample_accuracy.png')
    plt.close()
    plt.gcf().clear()
    plt.close()
    
    batch_time_taken.append(np.mean(bt.batch_time))

# Plot the time taken
batch_time_taken = np.array(batch_time_taken) * 1000
plt.figure()
plt.plot(neuron_nums, batch_time_taken, label='time_taken')
plt.xlabel('neuron_nums')
plt.xticks(neuron_nums)
plt.ylabel('time_taken(ms)')
plt.title('time for a weight update')
plt.savefig(plot_pic_folder + '3/p1a_batch_time.png')
# plt.show()
plt.gcf().clear()
plt.close()

total_time_taken = np.array(total_time_taken) * 1000
plt.figure()
plt.plot(neuron_nums, total_time_taken, label='time_taken')
plt.xlabel('neuron_nums')
plt.xticks(neuron_nums)
plt.ylabel('time_taken(ms)')
plt.title('total time for training')
plt.savefig(plot_pic_folder + '3/p1a_total_time.png')
# plt.show()
plt.gcf().clear()
plt.close()


# In[ ]:

#question 4

learning_rate = 0.01
epoch_num = 2

batch_time_taken = []
total_time_taken = []

decay_nums = [0,math.pow(10,-3),math.pow(10,-6),math.pow(10,-9),math.pow(10,-9)]
test_accuracy = []
for decay in decay_nums:
    bt = BatchTime()
    start_time = time.time()
    inputs = Input(shape=(trainX.shape[1],))
    dense1 = Dense(10, activation='sigmoid')(inputs)
    predictions = Dense(trainY.shape[1], activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=learning_rate, decay=decay)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    start_time = time.time()
    train_his = model.fit(trainX, trainY, epochs=epoch_num, verbose=2, 
                          validation_data=[testX,testY],
                          callbacks=[bt],
                          batch_size = 32)

    total_time_taken.append(time.time() - start_time)
#     loss, acc = model.evaluate(testX, testY, verbose=2)

    
    # Plot the training error against number of epochs
    plt.figure()
    plt.plot(range(1, 1+epoch_num), train_his.history['loss'], label='train_cost')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    annot_min(range(1, 1+epoch_num), np.array(train_his.history['loss']))
    plt.savefig(plot_pic_folder + '4/p1a_decay' + str(decay) + '_cost.png')
    plt.gcf().clear()
    plt.close()
    
    test_accuracy.append(max(train_his.history['val_acc']))
    batch_time_taken.append(np.mean(bt.batch_time))
    
#Plot the test accuracy against the different values of decay parameter.
plt.figure()
plt.plot(decay_nums, test_accuracy, label='test accuracy')
plt.xlabel('decay_nums')
plt.ylabel('accuracy')
plt.title('test accuracy')
annot_max(decay_nums, np.array(test_accuracy))
plt.savefig(plot_pic_folder + '4/p1a_test_accuracy.png')
plt.close()
plt.gcf().clear()
plt.close()
    

# Plot the time taken
batch_time_taken = np.array(batch_time_taken) * 1000
plt.figure()
plt.plot(decay_nums, batch_time_taken, label='time_taken')
plt.xlabel('decay_nums')
plt.xticks(decay_nums)
plt.ylabel('time_taken(ms)')
plt.title('time for a weight update')
plt.savefig(plot_pic_folder + '4/p1a_batch_time.png')
# plt.show()
plt.gcf().clear()
plt.close()

total_time_taken = np.array(total_time_taken) * 1000
plt.figure()
plt.plot(decay_nums, total_time_taken, label='time_taken')
plt.xlabel('decay_nums')
plt.xticks(decay_nums)
plt.ylabel('time_taken(ms)')
plt.title('total time for training')
plt.savefig(plot_pic_folder + '4/p1a_total_time.png')
# plt.show()
plt.gcf().clear()
plt.close()


# In[ ]:

# Question 5
# design a 4-layer network with two hidden- layers, 
# each consisting of 10 neurons with logistic activation functions, 
# batch size of 32 and decay parameter 10-6.


learning_rate = 0.01
epoch_num = 1000
decay = math.pow(10,-6)
batch_time_taken = []
total_time_taken = []

bt = BatchTime()
start_time = time.time()
inputs = Input(shape=(trainX.shape[1],))
dense1 = Dense(10, activation='sigmoid')(inputs)
dense2 = Dense(10, activation='sigmoid')(dense1)
predictions = Dense(trainY.shape[1], activation='softmax')(dense2)

model = Model(inputs=inputs, outputs=predictions)
sgd = optimizers.SGD(lr=learning_rate, decay=decay)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
start_time = time.time()
train_his = model.fit(trainX, trainY, epochs=epoch_num, verbose=2, 
                      validation_data=[testX,testY],
                      callbacks=[bt],
                      batch_size = 32)

print("Time taken to train a 4-layer network is : " + str(time.time() - start_time))
    
#  Plot the train and test accuracy of the 4-layer network.
# Plot the training error and the test accuracy against number of epochs
plt.figure()
plt.plot(range(1, 1+epoch_num), train_his.history['loss'], label='train_cost')
plt.xlabel('iterations')
plt.ylabel('cross-entropy')
plt.title('training cost')
annot_min(range(1, 1+epoch_num), np.array(train_his.history['loss']))
plt.savefig(plot_pic_folder + '5/p1a_cost.png')
plt.gcf().clear()
plt.close()


plt.figure()
plt.plot(range(1, 1+epoch_num), train_his.history['val_acc'], label='test_accuracy')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy')
annot_max(range(1, 1+epoch_num), np.array(train_his.history['val_acc']))
plt.savefig(plot_pic_folder + '5/p1a_sample_accuracy.png')
plt.close()
plt.gcf().clear()
plt.close()
# b) Compare and comment on the performances on 3-layer and 4-layer networks.


# In[ ]:




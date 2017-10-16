
# coding: utf-8

# In[ ]:

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time


# In[ ]:

def init_bias(n = 1):
    """
    Initialize the value of bias
    """
    return(theano.shared(np.zeros(n), theano.config.floatX))

def init_weights(n_in=1, n_out=1, logistic=True):
    """
    Initialize the value of weights
    """
    W_values = np.asarray(
        np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)),
        dtype=theano.config.floatX
        )
    if logistic == True:
        W_values *= 4
    return (theano.shared(value=W_values, name='W', borrow=True))


# In[ ]:

# scale data
X_min = None
X_max = None
def scale(X):
    """
    Min-Max Normalization
    """
    return (X - X_min)/(X_max-np.min(X, axis=0))

# def scale(X, X_min, X_max):
#     return (X - X_min)/(X_max-np.min(X, axis=0))


def sgd(cost, params, lr=0.01):
    """
    update parameters
    Stochastic Gradient Descent
    """
    grads = T.grad(cost=cost, wrt=params) # compute the gradient of cost 
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def read_data(filename):
    """
    Reading the file and return scaled features and labels
    """
    input_data = np.loadtxt(filename,delimiter=' ')
    X, _Y = input_data[:,:36], input_data[:,-1].astype(int)
#     X = scale(X, np.min(X, axis=0), np.max(X, axis=0))

    
    _Y[_Y == 7] = 6
    Y = np.zeros((_Y.shape[0], 6))
    Y[np.arange(_Y.shape[0]), _Y-1] = 1
    
    return X, Y


# In[ ]:

def train_network(trainX, trainY, testX, testY, decay, learning_rate, epochs, batch_size, num_neurons):   
    """
    Struct the model with 1 hidden layer
    and train the model
    """
    # theano expressions
    X = T.matrix() #features
    Y = T.matrix() #output

    w1, b1 = init_weights(36, num_neurons), init_bias(num_neurons) #weights and biases from input to hidden layer
    w2, b2 = init_weights(num_neurons, 6, logistic=False), init_bias(6) #weights and biases from hidden to output layer

#     activation
    h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
    py = T.nnet.softmax(T.dot(h1, w2) + b2)

    y_x = T.argmax(py, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
    params = [w1, b1, w2, b2]
    updates = sgd(cost, params, learning_rate)

    # compile
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
    
    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    batch_time_used = []
    
    for i in range(epochs):
        if i % 100 == 0:
            print(i)

        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            start_time = time.time()
            cost += train(trainX[start:end], trainY[start:end])
            batch_time_used.append(time.time()*1000-start_time*1000)
        
        
        train_cost = np.append(train_cost, cost/(n // batch_size))

        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))

    print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))

    return train_cost, test_accuracy, np.mean(batch_time_used), np.sum(batch_time_used)


# In[ ]:

def train_4layers(trainX, trainY, testX, testY, decay, learning_rate, epochs, batch_size, num_neurons): 
    """
    Struct the model with 2 hidden layers
    and train the model
    """
    # theano expressions
    X = T.matrix() #features
    Y = T.matrix() #output

    w1, b1 = init_weights(36, num_neurons), init_bias(num_neurons) #weights and biases from input to hidden layer
    w2, b2 = init_weights(num_neurons, num_neurons), init_bias(num_neurons)
    w3, b3 = init_weights(num_neurons, 6, logistic=False), init_bias(6) #weights and biases from hidden to output layer

    # activation
    h1 = T.nnet.sigmoid(T.dot(X, w1) + b1) 
    h2 = T.nnet.sigmoid(T.dot(h1, w2) + b2)
    py = T.nnet.softmax(T.dot(h2, w3) + b3)

    y_x = T.argmax(py, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2)))) + decay*(T.sum(T.sqr(w2)+T.sum(T.sqr(w3))))
    params = [w1, b1, w2, b2, w3, b3]
    updates = sgd(cost, params, learning_rate)

    # compile
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
    
    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    batch_time_used = []
    
    for i in range(epochs):
        if i % 100 == 0:
            print(i)

        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            start_time = time.time()
            cost += train(trainX[start:end], trainY[start:end])
            batch_time_used.append(time.time()*1000-start_time*1000)
        
        
        train_cost = np.append(train_cost, cost/(n // batch_size))

        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))

    print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))

    return train_cost, test_accuracy, np.mean(batch_time_used), np.sum(batch_time_used)


# In[ ]:

# Point out the maximum and minimum point on the graph

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
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.1), **kw)

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
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.9), **kw)

# This is for plot train errors and test accuracies against epochs
def plot1(filename, train_cost, test_accuracy, epochs=1000): 

    #Plots
    plt.figure()
    plt.plot(range(epochs), train_cost)
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    annot_min(range(epochs), train_cost)
    plt.savefig(filename + '_cost.png')

    plt.figure()
    plt.plot(range(epochs), test_accuracy)
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    annot_max(range(epochs), test_accuracy)
    plt.savefig(filename + '_accuracy.png')
    

#This is for plot time against parameters
def plot2(filename, parameters, parameter_name, updatetimes, totaltimes):
    # update time
    plt.figure()
    plt.plot(parameters, updatetimes)
    plt.xlabel(parameter_name)
    plt.ylabel('time in ms')
    plt.title('time for a weight update')
    plt.savefig(filename + '_update_time.png')
    
    # total time
    plt.figure()
    plt.plot(parameters, totaltimes)
    plt.xlabel(parameter_name)
    plt.ylabel('time in s')
    plt.title('total time for training')
    plt.savefig(filename + '_total_time.png')
    
def histogram(x, y, para_name, file_name, ylabel, title):
    # plot the histogram
    fig, ax = plt.subplots(figsize=(10,6))
    rect = ax.bar([0,1,2,3,4],y,align='center') # A bar chart
   
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    autolabel(rect)
    plt.xlabel(para_name)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(file_name)
    
#This is for plot time against parameters in bar chart(decay)
def plot_bar(filename, parameters, parameter_name, updatetimes, totaltimes):

    
    # update time
    histogram(parameters, updatetimes, parameter_name, filename + '_update_time.png', 'time in ms', 'time for a weight update')
    
    # total time
    histogram(parameters, totaltimes, parameter_name, filename + '_total_time.png', 'time in s', 'total time for training')


# In[ ]:




# In[ ]:

# Prepare Data
trainX, trainY = read_data('sat_train.txt')
testX, testY = read_data('sat_test.txt')

# Use min max value of traning data to do scaling
X_min = np.min(trainX, axis=0)
X_max = np.max(trainX, axis=0)

trainX = scale(trainX)
testX = scale(testX)
    
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)


# In[ ]:

# Question 1

decay = 1e-6
learning_rate = 0.01
epochs = 1000
num_neurons = 10
batch_size = 32

train_cost, test_accuracy, update_time, total_time = train_network(trainX, trainY, testX, testY, decay, learning_rate, epochs, batch_size, num_neurons)
plot1(filename='./graph_theano/1/question1', train_cost=train_cost, test_accuracy=test_accuracy)


# In[ ]:

# Question 2

decay = 1e-6
learning_rate = 0.01
epochs = 1000
num_neurons = 10
batches = [4, 8, 16, 32, 64]

updateTimes = []
totalTimes = []

for batch_size in batches:
    print("Running batch size: "+str(batch_size))
    train_cost, test_accuracy, update_time, total_time = train_network(trainX, trainY, testX, testY, decay, learning_rate, epochs, batch_size, num_neurons)
    updateTimes.append(update_time)
    totalTimes.append(total_time/1000)
    
    print("Taken average " + str(update_time) + " ms to update weight once")
    print("Taken " + str(total_time/1000) + " s total time")
    plot1(filename='./graph_theano/2/batch_'+str(batch_size), train_cost=train_cost, test_accuracy=test_accuracy)

plot2('./graph_theano/2/batch', batches, 'batch size', updateTimes, totalTimes)


# In[ ]:

# Question 3

decay = 1e-6
learning_rate = 0.01
epochs = 1000
nums_neurons = [5, 10, 15, 20, 25]
batch_size = 4

updateTimes = []
totalTimes = []

for num_neurons in nums_neurons:
    print("Running num of neurons: "+str(num_neurons))
    train_cost, test_accuracy, update_time, total_time = train_network(trainX, trainY, testX, testY, decay, learning_rate, epochs, batch_size, num_neurons)
    updateTimes.append(update_time)
    totalTimes.append(total_time/1000)
    
    print("Taken average " + str(update_time) + " ms to update weight once")
    print("Taken " + str(total_time/1000) + " s total time")
    plot1(filename='./graph_theano/3/neurons_'+str(num_neurons), train_cost=train_cost, test_accuracy=test_accuracy)

plot2('./graph_theano/3/neurons', nums_neurons, 'number of neurons', updateTimes, totalTimes)


# In[ ]:

# Question 4

decays = [0, 1e-3, 1e-6, 1e-9, 1e-12]
learning_rate = 0.01
epochs = 1000
num_neurons = 15
batch_size = 4

updateTimes = []
totalTimes = []

for decay in decays:
    print("Running decay: "+str(decay))
    train_cost, test_accuracy, update_time, total_time = train_network(trainX, trainY, testX, testY, decay, learning_rate, epochs, batch_size, num_neurons)
    updateTimes.append(update_time)
    totalTimes.append(total_time/1000)
    
    print("Taken average " + str(update_time) + " ms to update weight once")
    print("Taken " + str(total_time/1000) + " s total time")
    plot1(filename='./graph_theano/4/decay_'+str(decay), train_cost=train_cost, test_accuracy=test_accuracy, epochs=epochs)

plot_bar('./graph_theano/4/decay_', decays, 'decay', updateTimes, totalTimes)


# In[ ]:

# Question 5

decay = 1e-6
learning_rate = 0.01
epochs = 1000
num_neurons = 10
batch_size = 32

updateTimes = []
totalTimes = []

print("Running 4 layers...")
train_cost, test_accuracy, update_time, total_time = train_4layers(trainX, trainY, testX, testY, decays, learning_rate, epochs, batch_size, num_neurons)
updateTimes.append(update_time)
totalTimes.append(total_time/1000)

print("Taken average " + str(update_time) + " ms to update weight once")
print("Taken " + str(total_time/1000) + " s total time")
plot1(filename='./graph_theano/5/4layers', train_cost=train_cost, test_accuracy=test_accuracy)


# In[ ]:





# coding: utf-8

# In[ ]:


import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

np.random.seed(10)


# In[ ]:


# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)
 
def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

# update parameters
def sgd(cost, params, lr):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


# In[ ]:


#read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)


#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

# scale and normalize data
trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

trainX = scale(trainX, trainX_min, trainX_max)
testX = scale(testX, trainX_min, trainX_max)

# trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
# testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

# trainX = normalize(trainX, trainX_mean, trainX_std)
# testX = normalize(testX, testX_mean, testX_std)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)


# In[ ]:


def train_network(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, plot_filename):
    
    floatX = theano.config.floatX

    no_features = trainX.shape[1] 
    x = T.matrix('x') # data sample
    d = T.matrix('d') # desired output
    no_samples = T.scalar('no_samples')

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX ) 
    b_o = theano.shared(np.random.randn()*.01, floatX)
    w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
    b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

    # learning rate
    alpha = theano.shared(learning_rate, floatX) 


    #Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    y = T.dot(h1_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    #define gradients
    dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

#     params = [w_o, b_o, w_h1, b_h1]
#     updates = sgd(cost, params, alpha)
    
    
    train = theano.function(
            inputs = [x, d],
            outputs = cost,
            updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
            allow_input_downcast=True
            )
#     train = theano.function(
#         inputs = [x, d],
#         outputs = cost,
#         updates = updates,
#         allow_input_downcast=True
#         )

    test = theano.function(
        inputs = [x, d],
        outputs = [y, cost, accuracy],
        allow_input_downcast=True
        )


    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    min_error = 1e+15
    best_iter = 0
    best_w_o = np.zeros(no_hidden1)
    best_w_h1 = np.zeros([no_features, no_hidden1])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden1)

    alpha.set_value(learning_rate)
    print(alpha.get_value())


    # Training
    val_itr = 0
    t = time.time()

    val_accuracy = np.zeros(epochs)
    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    for iter in range(epochs):
        if iter % 100 == 0:
            print(iter)

        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        n = len(trainX)
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], np.transpose(trainY[start:end]))
#         train_cost[iter] = train(trainX, np.transpose(trainY))
        train_cost[iter] = cost/(n // batch_size)
        pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

        if test_cost[iter] < min_error:
            best_iter = iter
            min_error = test_cost[iter]
            best_w_o = w_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()

    #set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)

    best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))
    
    plot_train_error(plot_filename, train_cost, epochs)
    plot_test_error(plot_filename, test_cost, epochs)
    plot_test_accuracy(plot_filename, test_accuracy, epochs)


# In[ ]:


def train_network_validation(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, plot_filename):
    
    floatX = theano.config.floatX

    no_features = trainX.shape[1] 
    x = T.matrix('x') # data sample
    d = T.matrix('d') # desired output
    no_samples = T.scalar('no_samples')

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX ) 
    b_o = theano.shared(np.random.randn()*.01, floatX)
    w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
    b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

    # learning rate
    alpha = theano.shared(learning_rate, floatX) 


    #Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    y = T.dot(h1_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    #define gradients

    dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

    
    
    train = theano.function(
            inputs = [x, d],
            outputs = cost,
            updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
            allow_input_downcast=True
            )

    test = theano.function(
        inputs = [x, d],
        outputs = [y, cost, accuracy],
        allow_input_downcast=True
        )



    # Training
    kf = KFold(n_splits=5)
    val_itr = 0
    t = time.time()
    
    train_costs = []
    test_costs = []
    val_costs = []
    
    for train_index, val_index in kf.split(trainX):

        train_cost = np.zeros(epochs)
        val_cost = np.zeros(epochs)
        val_accuracy = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        
        train_cost = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        
        w_o.set_value(np.random.randn(no_hidden1)*.01)
        b_o.set_value(np.random.randn()*.01)
        w_h1.set_value(np.random.randn(no_features, no_hidden1)*.01)
        b_h1.set_value(np.random.randn(no_hidden1)*.01)

        min_error = 1e+15
        best_iter = 0
        best_w_o = np.zeros(no_hidden1)
        best_w_h1 = np.zeros([no_features, no_hidden1])
        best_b_o = 0
        best_b_h1 = np.zeros(no_hidden1)

        alpha.set_value(learning_rate)
        print(alpha.get_value())

        val_itr += 1
        print("k fold validation: " + str(val_itr))
        print("TRAIN: "+str(train_index) + " VALID: "+str(val_index))
        
        val_set_X = trainX[val_index]
        val_set_Y = trainY[val_index]
        train_set_X = trainX[train_index]
        train_set_Y = trainY[train_index]

        for iter in range(epochs):
            if iter % 100 == 0:
                print(iter)

            trainX, trainY = shuffle_data(trainX, trainY)
#             cost = 0.0
#             n = len(train_set_X)
#             for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
#                 cost += train(train_set_X[start:end], np.transpose(train_set_Y[start:end]))
            
#             train_cost[iter] = cost/(n // batch_size)
            train_cost[iter] = train(trainX, np.transpose(trainY))
            val_pred, val_cost[iter], val_accuracy[iter] = test(val_set_X, np.transpose(val_set_Y))
            pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

            if test_cost[iter] < min_error:
                best_iter = iter
                min_error = test_cost[iter]
                best_w_o = w_o.get_value()
                best_w_h1 = w_h1.get_value()
                best_b_o = b_o.get_value()
                best_b_h1 = b_h1.get_value()

        #set weights and biases to values at which performance was best
        w_o.set_value(best_w_o)
        b_o.set_value(best_b_o)
        w_h1.set_value(best_w_h1)
        b_h1.set_value(best_b_h1)
        
        test_costs.append(test_cost)
        train_costs.append(train_cost)
        val_costs.append(val_cost)
        

        best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

        print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))
        

        
    plot_train_val_error(plot_filename, np.mean(train_costs, axis=0), np.mean(val_costs, axis=0), epochs)
    plot_test_error(plot_filename, np.mean(test_costs, axis=0), epochs)


# In[ ]:


def train_4layers_validation(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, no_hidden2, plot_filename):
    
    floatX = theano.config.floatX

    no_features = trainX.shape[1] 
    x = T.matrix('x') # data sample
    d = T.matrix('d') # desired output
    no_samples = T.scalar('no_samples')

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden2)*.01, floatX ) 
    b_o = theano.shared(np.random.randn()*.01, floatX)
    w_h2 = theano.shared(np.random.randn(no_hidden1, no_hidden2)*.01, floatX )
    b_h2 = theano.shared(np.random.randn(no_hidden2)*0.01, floatX)
    w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
    b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)
 

    # learning rate
    alpha = theano.shared(learning_rate, floatX) 


    #Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
    y = T.dot(h2_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    #define gradients

    dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2])

    
    
    train = theano.function(
            inputs = [x, d],
            outputs = cost,
            updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h1],
                   [b_h1, b_h1 - alpha*db_h1],
                   [w_h2, w_h2 - alpha*dw_h2],
                   [b_h2, b_h2 - alpha*db_h2]],
            allow_input_downcast=True
            )

    test = theano.function(
        inputs = [x, d],
        outputs = [y, cost, accuracy],
        allow_input_downcast=True
        )



    # Training
    kf = KFold(n_splits=5)
    val_itr = 0
    t = time.time()
    
    train_costs = []
    test_costs = []
    val_costs = []
    
    for train_index, val_index in kf.split(trainX):

        train_cost = np.zeros(epochs)
        val_cost = np.zeros(epochs)
        val_accuracy = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        
        train_cost = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        
        w_o.set_value(np.random.randn(no_hidden2)*.01)
        b_o.set_value(np.random.randn()*.01)
        w_h2.set_value(np.random.randn(no_hidden1, no_hidden2)*.01)
        b_h2.set_value(np.random.randn(no_hidden2)*.01)
        w_h1.set_value(np.random.randn(no_features, no_hidden1)*.01)
        b_h1.set_value(np.random.randn(no_hidden1)*.01)


        min_error = 1e+15
        best_iter = 0
        best_w_o = np.zeros(no_hidden1)
        best_w_h1 = np.zeros([no_features, no_hidden1])
        best_b_o = 0
        best_b_h1 = np.zeros(no_hidden1)

        alpha.set_value(learning_rate)
        print(alpha.get_value())

        val_itr += 1
        print("k fold validation: " + str(val_itr))
        print("TRAIN: "+str(train_index) + " VALID: "+str(val_index))
        
        val_set_X = trainX[val_index]
        val_set_Y = trainY[val_index]
        train_set_X = trainX[train_index]
        train_set_Y = trainY[train_index]

        for iter in range(epochs):
            if iter % 100 == 0:
                print(iter)

            trainX, trainY = shuffle_data(trainX, trainY)
#             cost = 0.0
#             n = len(train_set_X)
#             for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
#                 cost += train(train_set_X[start:end], np.transpose(train_set_Y[start:end]))
            
#             train_cost[iter] = cost/(n // batch_size)
            train_cost[iter] = train(trainX, np.transpose(trainY))
            val_pred, val_cost[iter], val_accuracy[iter] = test(val_set_X, np.transpose(val_set_Y))
            pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

            if test_cost[iter] < min_error:
                best_iter = iter
                min_error = test_cost[iter]
                best_w_o = w_o.get_value()
                best_w_h1 = w_h1.get_value()
                best_b_o = b_o.get_value()
                best_b_h1 = b_h1.get_value()

        #set weights and biases to values at which performance was best
        w_o.set_value(best_w_o)
        b_o.set_value(best_b_o)
        w_h1.set_value(best_w_h1)
        b_h1.set_value(best_b_h1)
        
        test_costs.append(test_cost)
        train_costs.append(train_cost)
        val_costs.append(val_cost)
        

        best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

        print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))
        

        
    plot_train_val_error(plot_filename, np.mean(train_costs, axis=0), np.mean(val_costs, axis=0), epochs)
    plot_test_error(plot_filename, np.mean(test_costs, axis=0), epochs)


# In[ ]:


def train_5layers_validation(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, no_hidden2, no_hidden3, plot_filename):
    
    floatX = theano.config.floatX

    no_features = trainX.shape[1] 
    x = T.matrix('x') # data sample
    d = T.matrix('d') # desired output
    no_samples = T.scalar('no_samples')

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden3)*.01, floatX ) 
    b_o = theano.shared(np.random.randn()*.01, floatX)
    w_h3 = theano.shared(np.random.randn(no_hidden2, no_hidden3)*.01, floatX )
    b_h3 = theano.shared(np.random.randn(no_hidden3)*0.01, floatX)
    w_h2 = theano.shared(np.random.randn(no_hidden1, no_hidden2)*.01, floatX )
    b_h2 = theano.shared(np.random.randn(no_hidden2)*0.01, floatX)
    w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
    b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)
 

    # learning rate
    alpha = theano.shared(learning_rate, floatX) 


    #Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
    h3_out = T.nnet.sigmoid(T.dot(h2_out, w_h3) + b_h3)
    y = T.dot(h3_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    #define gradients

    dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2, dw_h3, db_h3 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2, w_h3, b_h3])

    
    
    train = theano.function(
            inputs = [x, d],
            outputs = cost,
            updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h1],
                   [b_h1, b_h1 - alpha*db_h1],
                   [w_h2, w_h2 - alpha*dw_h2],
                   [b_h2, b_h2 - alpha*db_h2],
                   [w_h3, w_h3 - alpha*dw_h3],
                   [b_h3, b_h3 - alpha*db_h3]],
            allow_input_downcast=True
            )

    test = theano.function(
        inputs = [x, d],
        outputs = [y, cost, accuracy],
        allow_input_downcast=True
        )



    # Training
    kf = KFold(n_splits=5)
    val_itr = 0
    t = time.time()
    
    train_costs = []
    test_costs = []
    val_costs = []
    
    for train_index, val_index in kf.split(trainX):

        train_cost = np.zeros(epochs)
        val_cost = np.zeros(epochs)
        val_accuracy = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        
        train_cost = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        
        w_o.set_value(np.random.randn(no_hidden3)*.01)
        b_o.set_value(np.random.randn()*.01)
        w_h3.set_value(np.random.randn(no_hidden2, no_hidden3)*.01)
        b_h3.set_value(np.random.randn(no_hidden3)*.01)
        w_h2.set_value(np.random.randn(no_hidden1, no_hidden2)*.01)
        b_h2.set_value(np.random.randn(no_hidden2)*.01)
        w_h1.set_value(np.random.randn(no_features, no_hidden1)*.01)
        b_h1.set_value(np.random.randn(no_hidden1)*.01)


        min_error = 1e+15
        best_iter = 0
        best_w_o = np.zeros(no_hidden1)
        best_w_h1 = np.zeros([no_features, no_hidden1])
        best_b_o = 0
        best_b_h1 = np.zeros(no_hidden1)

        alpha.set_value(learning_rate)
        print(alpha.get_value())

        val_itr += 1
        print("k fold validation: " + str(val_itr))
        print("TRAIN: "+str(train_index) + " VALID: "+str(val_index))
        
        val_set_X = trainX[val_index]
        val_set_Y = trainY[val_index]
        train_set_X = trainX[train_index]
        train_set_Y = trainY[train_index]

        for iter in range(epochs):
            if iter % 100 == 0:
                print(iter)

            trainX, trainY = shuffle_data(trainX, trainY)
#             cost = 0.0
#             n = len(train_set_X)
#             for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
#                 cost += train(train_set_X[start:end], np.transpose(train_set_Y[start:end]))
            
#             train_cost[iter] = cost/(n // batch_size)
            train_cost[iter] = train(trainX, np.transpose(trainY))
            val_pred, val_cost[iter], val_accuracy[iter] = test(val_set_X, np.transpose(val_set_Y))
            pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

            if test_cost[iter] < min_error:
                best_iter = iter
                min_error = test_cost[iter]
                best_w_o = w_o.get_value()
                best_w_h1 = w_h1.get_value()
                best_b_o = b_o.get_value()
                best_b_h1 = b_h1.get_value()

        #set weights and biases to values at which performance was best
        w_o.set_value(best_w_o)
        b_o.set_value(best_b_o)
        w_h1.set_value(best_w_h1)
        b_h1.set_value(best_b_h1)
        
        test_costs.append(test_cost)
        train_costs.append(train_cost)
        val_costs.append(val_cost)
        

        best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

        print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))
        

        
    plot_train_val_error(plot_filename, np.mean(train_costs, axis=0), np.mean(val_costs, axis=0), epochs)
    plot_test_error(plot_filename, np.mean(test_costs, axis=0), epochs)


# In[ ]:


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
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="bottom")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.9), **kw)

def annot_min2(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "MIN Point x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="bottom")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.7), **kw)
    
    
def plot_train_error(filename_prefix, train_cost, epochs=1000):
    plt.figure()
    plt.plot(range(epochs), train_cost, label='train error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Errors')
#     plt.legend()
    annot_min(range(epochs), train_cost)
    plt.savefig(filename_prefix + '_train_error.png')
    
def plot_test_error(filename_prefix, test_cost, epochs=1000):
    plt.figure()
    plt.plot(range(epochs), test_cost, label='train error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Errors')
#     plt.legend()
    annot_min(range(epochs), test_cost)
    plt.savefig(filename_prefix + '_test_error.png')
    
def plot_test_accuracy(filename_prefix, test_accuracy, epochs=1000):
    plt.figure()
    plt.plot(range(epochs), test_accuracy, label='test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
#     plt.legend()
    annot_min(range(epochs), test_accuracy)
    plt.savefig(filename_prefix + '_test_accuracy.png') 

def plot_train_val_error(filename_prefix, train_cost, val_cost, epochs=1000):
    plt.figure()
    plt.plot(range(epochs), train_cost, label='train error')
    plt.plot(range(epochs), val_cost, label='validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Validation Errors')
#     annot_min(range(epochs),train_cost)
#     annot_min2(range(epochs),val_cost)
    plt.legend()
    plt.savefig(filename_prefix + '_train_val_error.png')


# In[ ]:


#Question 1

epochs = 1000
batch_size = 64
no_hidden1 = 30 #num of neurons in hidden layer 1
learning_rate = 1e-4

train_network(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, './theano_graph/1/Question1')


# In[ ]:


# Question 2

epochs = 1000
batch_size = 32
no_hidden1 = 30 #num of neurons in hidden layer 1
learning_rates = [1e-3, 0.5e-3, 1e-4, 0.5e-4, 1e-5]

for learning_rate in learning_rates:
    print("Running Learning Rate: "+ str(learning_rate))
    train_network_validation(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, './theano_graph/2/learning_rate_'+str(learning_rate))


# In[ ]:


# Question 3

epochs = 1000
batch_size = 32
nos_hidden1 = [20, 30, 40, 50 ,60] #num of neurons in hidden layer 1
learning_rate = 1e-3

for no_hidden1 in nos_hidden1:
    print("Running Num of Hidden 1: "+ str(no_hidden1))
    train_network_validation(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, './theano_graph/3/no_hidden1_'+str(no_hidden1))


# In[ ]:


# Question 4
epochs = 1000
batch_size = 32
no_hidden1 = 60 #num of neurons in hidden layer 1
no_hidden2 = 20
no_hidden3 = 20
learning_rate = 1e-4

train_4layers_validation(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, no_hidden2, './theano_graph/4/layers_4')
train_5layers_validation(trainX, trainY, testX, testY, learning_rate, epochs, batch_size, no_hidden1, no_hidden2, no_hidden3, './theano_graph/4/layers_5')


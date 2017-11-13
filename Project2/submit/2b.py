
# coding: utf-8

# In[ ]:

from load import mnist
import numpy as np

import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# In[ ]:

# 1 encoder, decoder and a softmax layer

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)


# In[ ]:

def plot_mnist_data(X, file_name):
    pylab.figure()
    pylab.gray()
    size = int(np.sqrt(X[0].shape[0]))
    for i in range(100):
        pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(X[i,:].reshape(size,size))
    pylab.savefig('./Graph/' + file_name)
    pylab.close()
    


# In[ ]:

def plot_weight(weight, tag):
    # Plot 100 samples of weights (as images) learned at each layer
    w = weight.get_value()
    pylab.figure()
    pylab.gray()
    size = int(np.sqrt(w.shape[0]))
    for i in range(100):
        pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w[:,i].reshape(size,size))
    pylab.savefig('./Graph/' + tag + '_weight.png')
    pylab.close()
    
    print('plot_weight finished!')


# In[ ]:

def plot_training_error(d, tag):
    global training_epochs
    pylab.figure()
    pylab.plot(range(training_epochs), d)
    pylab.xlabel('iterations')
    pylab.ylabel('cross-entropy training error')
    pylab.savefig('./Graph/' + tag + '_training_error.png')
    pylab.close()
    
def plot_test_accuracy(acc, tag):
    global training_epochs
    pylab.figure()
    pylab.plot(range(training_epochs), acc)
    pylab.xlabel('iterations')
    pylab.ylabel('test accuracy')
    pylab.savefig('./Graph/' + tag + '_test_acc.png')
    pylab.close()


# In[ ]:

# load data
trX, teX, trY, teY = mnist()

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

print(trX.shape)


# In[ ]:

# hyper-parameter
corruption_level=0.1
learning_rate = 0.1

momentum = 0.1
beta = 0.5 # penalty parameter
rho = 0.05 # sparcity parameter


# In[ ]:

# question B (1) & B(2)
# construct the network
def construct_nn_part1_2():
    x = T.fmatrix('x')  
    d = T.fmatrix('d')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    no_hidden1 = 900
    no_hidden2 = 625
    no_hidden3 = 400


    W1, b1 = init_weights(28*28, no_hidden1) , init_bias(no_hidden1)
    W2, b2 = init_weights(no_hidden1, no_hidden2), init_bias(no_hidden2)
    W3, b3 = init_weights(no_hidden2, no_hidden3), init_bias(no_hidden3)
    W4, b4 = init_weights(no_hidden3, 10), init_bias(10) # output layer for question B(2)


    b1_prime = init_bias(28*28)
    W1_prime = W1.transpose() # (900,784)
    b2_prime = init_bias(no_hidden1)
    W2_prime = W2.transpose() # (625, 900)
    b3_prime = init_bias(no_hidden2)
    W3_prime = W3.transpose() # (400,625)



    #  train on the inputs tilde_x to learn primary features y1
    tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
    y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
    z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
    cost_da1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))
    params_da1 = [W1, b1, b1_prime]
    grads_da1 = T.grad(cost_da1, params_da1)
    updates_da1 = [(param_da, param_da - learning_rate * grad_da)
               for param_da, grad_da in zip(params_da1, grads_da1)]
    train_da1 = theano.function(inputs=[x], outputs = cost_da1, updates = updates_da1, allow_input_downcast = True)
    
    #  train on the inputs y1 to learn features y2
    tilde_y1 = theano_rng.binomial(size=y1.shape, n=1, p=1 - corruption_level,
                          dtype=theano.config.floatX)*y1
    y2 = T.nnet.sigmoid(T.dot(tilde_y1, W2) + b2)
    z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
    cost_da2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))
    params_da2 = [W2, b2, b2_prime]
    grads_da2 = T.grad(cost_da2, params_da2)
    updates_da2 = [(param_da, param_da - learning_rate * grad_da)
               for param_da, grad_da in zip(params_da2, grads_da2)]
    train_da2 = theano.function(inputs=[x], outputs = cost_da2, updates = updates_da2, allow_input_downcast = True)

    #  train on the inputs y2 to learn features y3
    tilde_y2 = theano_rng.binomial(size=y2.shape, n=1, p=1 - corruption_level,
                          dtype=theano.config.floatX)*y2
    y3 = T.nnet.sigmoid(T.dot(tilde_y2, W3) + b3)
    z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
    cost_da3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))
    params_da3 = [W3, b3, b3_prime]
    grads_da3 = T.grad(cost_da3, params_da3)
    updates_da3 = [(param_da, param_da - learning_rate * grad_da)
               for param_da, grad_da in zip(params_da3, grads_da3)]
    train_da3 = theano.function(inputs=[x], outputs = cost_da3, updates = updates_da3, allow_input_downcast = True)
    
    encoder1 = theano.function(inputs=[x], outputs = y1, allow_input_downcast=True)
    encoder2 = theano.function(inputs=[y1], outputs = y2, allow_input_downcast=True)
    encoder3 = theano.function(inputs=[y2], outputs = y3, allow_input_downcast=True)
    
    decoder3 = theano.function(inputs=[y3],outputs = z3, allow_input_downcast=True) # 625
    decoder2 = theano.function(inputs=[y2],outputs = z2, allow_input_downcast=True) # 900
    decoder1 = theano.function(inputs=[y1], outputs = z1, allow_input_downcast=True) # 784

    
    # five-layer feedforward neuron network
    output_ff = T.nnet.softmax(T.dot(y3, W4)+b4)
    predicted_result_ff = T.argmax(output_ff, axis=1)
    cost_ff = T.mean(T.nnet.categorical_crossentropy(output_ff, d))

    params_ff = [W1, b1, W2, b2, W3, b3, W4, b4]
    grads_ff = T.grad(cost_ff, params_ff)
    updates_ff = [(param_ff, param_ff - learning_rate * grad_ff)
               for param_ff, grad_ff in zip(params_ff, grads_ff)]
    train_ffn = theano.function(inputs=[x, d], outputs = cost_ff, updates = updates_ff, allow_input_downcast = True)
    test_ffn = theano.function(inputs=[x], outputs = predicted_result_ff, allow_input_downcast=True)
    
    return [train_da1, train_da2, train_da3], [encoder1, encoder2, encoder3], [decoder3, decoder2, decoder1], train_ffn, test_ffn, W1, W2, W3


# In[ ]:

# question B1
train_da, encoder, decoder, train_ffn, test_ffn, W1, W2, W3 = construct_nn_part1_2()
print('training dae1 ...')
training_epochs = 25
batch_size = 128
reconstruction_error = []

for i in range(3):
    d = []
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_da[i](trX[start:end])) # costs

        d.append(np.mean(c, dtype='float64')) # reconstruction errors
    
    reconstruction_error.append(d)
    print("Finished training layer %d" % (i+1))


# In[ ]:

id_list = list(range(len(teX)))
np.random.shuffle(id_list)
test_id = id_list[:100]


# In[ ]:

# result plotting for question B1
plot_mnist_data(trX, "1/1_original")
plot_mnist_data(teX[test_id], "1/1_test")
plot_weight(W1, '1/1_W1')
plot_weight(W2, '1/1_W2')  
plot_weight(W3, '1/1_W3')
encoded_image = teX[test_id]
for i in range(len(encoder)):
    encoded_image = encoder[i](encoded_image)
    plot_mnist_data(encoded_image, "1/1_" + str(i+1) + "rd_hidden_layer_activation")
decoded_image = encoded_image
for i in decoder:
    decoded_image = i(decoded_image)
plot_mnist_data(decoded_image, "1/test_reconstructed")

# plot learning curves (i.e., reconstruction errors on training data) for training each epoch
for i in range(len(reconstruction_error)):
    plot_training_error(reconstruction_error[i], '1/1_' + str(i+1) + 'rd_layer')


# In[ ]:

# question B2
print('\ntraining ffn ...')
ff_training_cost, ff_acc = [], []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_ffn(trX[start:end], trY[start:end]))
    ff_training_cost.append(np.mean(c, dtype='float64')) # training cost
    ff_acc.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX))) # accuracy
    print(ff_acc[epoch])


# In[ ]:

# result plotting for question B2
plot_training_error(ff_training_cost, '2/2')
plot_test_accuracy(ff_acc, '2/2')


# In[ ]:

# question B (3)
# construct the network
def sgd_momentum(cost, params, lr, momentum):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value())
#         v_new = momentum*v - (g + decay*p) * lr 
        v_new = momentum*v - g * lr 
        updates.append([p, p + v_new])
        updates.append([v, v_new])
        return updates
    
def construct_nn_part3():
    x = T.fmatrix('x')  
    d = T.fmatrix('d')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    no_hidden1 = 900
    no_hidden2 = 625
    no_hidden3 = 400

    W1, b1 = init_weights(28*28, no_hidden1) , init_bias(no_hidden1)
    W2, b2 = init_weights(no_hidden1, no_hidden2), init_bias(no_hidden2)
    W3, b3 = init_weights(no_hidden2, no_hidden3), init_bias(no_hidden3)
    W4, b4 = init_weights(no_hidden3, 10), init_bias(10) # output layer for question B(2)


    b1_prime = init_bias(28*28)
    W1_prime = W1.transpose() # (900,784)
    b2_prime = init_bias(no_hidden1)
    W2_prime = W2.transpose() # (625, 900)
    b3_prime = init_bias(no_hidden2)
    W3_prime = W3.transpose() # (400,625)

    #  train on the inputs tilde_x to learn primary features y1
    tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                                  dtype=theano.config.floatX)*x
    y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
    z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
    cost_da1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1)) 
    + beta*T.shape(y1)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) 
    - beta*rho*T.sum(T.log(T.mean(y1, axis=0)+1e-6)) 
    - beta*(1-rho)*T.sum(T.log(1-T.mean(y1, axis=0)+1e-6))
                
    params_da1 = [W1, b1, b1_prime]
    grads_da1 = T.grad(cost_da1, params_da1)
#     updates_da1 = [(param_da, momentum * param_da - learning_rate * grad_da)
#                for param_da, grad_da in zip(params_da1, grads_da1)]
    train_da1 = theano.function(inputs=[x], outputs = cost_da1, updates = sgd_momentum(cost_da1, params_da1, learning_rate, momentum), allow_input_downcast = True)
    
    #  train on the inputs y1 to learn features y2
    tilde_y1 = theano_rng.binomial(size=y1.shape, n=1, p=1 - corruption_level,
                                  dtype=theano.config.floatX)*y1
    y2 = T.nnet.sigmoid(T.dot(tilde_y1, W2) + b2)
    z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
    cost_da2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))
    + beta*T.shape(y2)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) 
    - beta*rho*T.sum(T.log(T.mean(y2, axis=0)+1e-6)) 
    - beta*(1-rho)*T.sum(T.log(1-T.mean(y2, axis=0)+1e-6))
                
    params_da2 = [W2, b2, b2_prime]
    grads_da2 = T.grad(cost_da2, params_da2)
#     updates_da2 = [(param_da, momentum * param_da - learning_rate * grad_da)
#                for param_da, grad_da in zip(params_da2, grads_da2)]
    train_da2 = theano.function(inputs=[x], outputs = cost_da2, updates = sgd_momentum(cost_da2, params_da2, learning_rate, momentum), allow_input_downcast = True)

    #  train on the inputs y2 to learn features y3
    tilde_y2 = theano_rng.binomial(size=y2.shape, n=1, p=1 - corruption_level,
                                  dtype=theano.config.floatX)*y2
    y3 = T.nnet.sigmoid(T.dot(tilde_y2, W3) + b3)
    z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
    cost_da3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))
    + beta*T.shape(y3)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) 
    - beta*rho*T.sum(T.log(T.mean(y3, axis=0)+1e-6)) 
    - beta*(1-rho)*T.sum(T.log(1-T.mean(y3, axis=0)+1e-6))
                
    params_da3 = [W3, b3, b3_prime]
    grads_da3 = T.grad(cost_da3, params_da3)
#     updates_da3 = [(param_da, momentum * param_da - learning_rate * grad_da)
#                for param_da, grad_da in zip(params_da3, grads_da3)]
    train_da3 = theano.function(inputs=[x], outputs = cost_da3, updates = sgd_momentum(cost_da3, params_da3, learning_rate, momentum), allow_input_downcast = True)
    
    encoder1 = theano.function(inputs=[x], outputs = y1, allow_input_downcast=True)
    encoder2 = theano.function(inputs=[y1], outputs = y2, allow_input_downcast=True)
    encoder3 = theano.function(inputs=[y2], outputs = y3, allow_input_downcast=True)
    
    decoder3 = theano.function(inputs=[y3],outputs = z3, allow_input_downcast=True) # 625
    decoder2 = theano.function(inputs=[y2],outputs = z2, allow_input_downcast=True) # 900
    decoder1 = theano.function(inputs=[y1], outputs = z1, allow_input_downcast=True) # 784

    
    # five-layer feedforward neuron network
    output_ff = T.nnet.softmax(T.dot(y3, W4)+b4)
    predicted_result_ff = T.argmax(output_ff, axis=1)
    cost_ff = T.mean(T.nnet.categorical_crossentropy(output_ff, d))

    params_ff = [W1, b1, W2, b2, W3, b3, W4, b4]
    grads_ff = T.grad(cost_ff, params_ff)
    train_ffn = theano.function(inputs=[x, d], outputs = cost_ff, updates = sgd_momentum(cost_ff, params_ff, learning_rate, momentum), allow_input_downcast = True)
    test_ffn = theano.function(inputs=[x], outputs = predicted_result_ff, allow_input_downcast=True)
    
    return [train_da1, train_da2, train_da3], [encoder1, encoder2, encoder3], [decoder3, decoder2, decoder1], train_ffn, test_ffn, W1, W2, W3


# In[ ]:

# Question B3
train_da, encoder, decoder, train_ffn, test_ffn, W1, W2, W3 = construct_nn_part3()
print('training dae1 ...')
training_epochs = 25
batch_size = 128
reconstruction_error = []

for i in range(3):
    d = []
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_da[i](trX[start:end])) # costs

        d.append(np.mean(c, dtype='float64')) # reconstruction errors
    
    reconstruction_error.append(d)
    print("Finished training layer %d" % (i+1))


# In[ ]:

id_list = list(range(len(teX)))
np.random.shuffle(id_list)
test_id = id_list[:100]


# In[ ]:

# result plotting for question B3
plot_mnist_data(trX, "3/1_original")
plot_mnist_data(teX[test_id], "3/1_test")
plot_weight(W1, '3/1_W1')
plot_weight(W2, '3/1_W2')  
plot_weight(W3, '3/1_W3')
encoded_image = teX[test_id]
for i in range(len(encoder)):
    encoded_image = encoder[i](encoded_image)
    plot_mnist_data(encoded_image, "3/1_" + str(i+1) + "rd_hidden_layer_activation")
decoded_image = encoded_image
for i in decoder:
    decoded_image = i(decoded_image)
plot_mnist_data(decoded_image, "3/test_reconstructed")

# plot learning curves (i.e., reconstruction errors on training data) for training each epoch
for i in range(len(reconstruction_error)):
    plot_training_error(reconstruction_error[i], '3/1_' + str(i+1) + 'rd_layer')


# In[ ]:

print('\ntraining ffn ...')
ff_training_cost, ff_acc = [], []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_ffn(trX[start:end], trY[start:end]))
    ff_training_cost.append(np.mean(c, dtype='float64')) # training cost
    ff_acc.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX))) # accuracy
    print(ff_acc[epoch])
    


# In[ ]:

# result plotting for question B2
plot_training_error(ff_training_cost, '3/2')
plot_test_accuracy(ff_acc, '3/2')


# In[ ]:




# In[ ]:




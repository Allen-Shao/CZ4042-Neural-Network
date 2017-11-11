from load import mnist
import numpy as np 
import pylab

import theano
from theano import tensor as T 
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

np.random.seed(10)


def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def model(X, w1, b1, w2, b2, w3, b3, w4, b4):
	# First convolution and pooling layer
	y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
	pool_dim1 = (2, 2)
	o1 = pool.pool_2d(y1, pool_dim1, ignore_border=True)

	# Second convolution and pooling layer
	y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
	pool_dim2 = (2, 2)
	o2 = pool.pool_2d(y2, pool_dim2, ignore_border=True)

	# FC layer
	fc = T.nnet.relu(T.dot(T.flatten(o2, outdim=2), w3) + b3)

	# Softmax
	pyx = T.nnet.softmax(T.dot(fc, w4) + b4)

	return y1, o1, y2, o2, pyx

def sgd(cost, params, lr=0.05, decay=0.0001):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - (g + decay*p) * lr])
    return updates

def shuffle_data(samples, labels):
	idx = np.arange(samples.shape[0])
	np.random.shuffle(idx)
	samples, labels = samples[idx], labels[idx]
	return samples, labels


trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

X = T.tensor4('X')
Y = T.matrix('Y')

batch_size = 128
learning_rate = 0.05
decay = 1e-4
epochs = 100

num_filters_1 = 15
num_filters_2 = 20
fc_size = 100
softmax_size = 10

w1, b1 = init_weights_bias4((num_filters_1, 1, 9, 9), X.dtype)
w2, b2 = init_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype)
w3, b3 = init_weights_bias2((num_filters_2*3*3, fc_size), X.dtype)
w4, b4 = init_weights_bias2((fc_size, softmax_size), X.dtype)

y1, o1, y2, o2, py_x = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]

updates = sgd(cost, params, lr=learning_rate, decay=decay)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test_conv1 = theano.function(inputs = [X], outputs=[y1, o1], allow_input_downcast=True)
test_conv2 = theano.function(inputs = [X], outputs=[y2, o2], allow_input_downcast=True)

a = []
costs = []
max_epoch = -1
max_accuracy = 0
for i in range(epochs):
	trX, trY = shuffle_data (trX, trY)
	teX, teY = shuffle_data (teX, teY)
	cost_value = 0
	for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
		cost_value += train(trX[start:end], trY[start:end])
	costs.append(cost_value/(len(trX)//batch_size))
	accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))
	if (accuracy > max_accuracy):
		max_accuracy = accuracy
		max_epoch = i
	a.append(accuracy)
	print('No.%3d Accuracy = %f'%(i+1,a[i]))

print('Max Accuracy = %f at iteration %d'%(max_accuracy, max_epoch+1))


pylab.figure()
pylab.plot(range(epochs), a)
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.savefig('figure_2a_1_test_accuracy.png')

pylab.figure()
pylab.plot(range(epochs), costs)
pylab.xlabel('epochs')
pylab.ylabel('training cost')
pylab.savefig('figure_2a_1_train_cost.png')

# w = w1.get_value()
# pylab.figure()
# pylab.gray()
# for i in range(15):
#     pylab.subplot(3, 5, i+1); pylab.axis('off'); pylab.imshow(w[i,:,:,:].reshape(9,9))
# #pylab.title('filters learned')
# pylab.savefig('figure_2a_2.png')

ind = np.random.randint(low=0, high=2000)

pylab.figure()
pylab.gray()
pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
pylab.savefig('input.png')

convolution1, pool1 = test_conv1(teX[ind:ind+1,:])
pylab.figure()
pylab.gray()
for i in range(15):
	pylab.subplot(3, 5, i+1); pylab.axis('off'); pylab.imshow(convolution1[0,i,:].reshape(20,20))
pylab.savefig('conv1.png')

pylab.figure()
pylab.gray()
for i in range(15):
	pylab.subplot(3, 5, i+1); pylab.axis('off'); pylab.imshow(pool1[0,i,:].reshape(10,10))
pylab.savefig('pool1.png')

convolution2, pool2 = test_conv2(teX[ind:ind+1,:])
pylab.figure()
pylab.gray()
for i in range(20):
	pylab.subplot(4, 5, i+1); pylab.axis('off'); pylab.imshow(convolution2[0,i,:].reshape(6,6))
pylab.savefig('conv2.png')

pylab.figure()
pylab.gray()
for i in range(20):
	pylab.subplot(4, 5, i+1); pylab.axis('off'); pylab.imshow(pool2[0,i,:].reshape(3,3))
pylab.savefig('pool2.png')
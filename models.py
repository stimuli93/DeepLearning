import layers
import numpy as np


class MLP_relu(object):

    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.W1 = np.random.random(size=(inputSize, hiddenSize)) - 0.5
        self.b1 = np.random.random(size=(hiddenSize)) - 0.5
        self.W2 = np.random.random(size=(hiddenSize, outputSize)) - 0.5
        self.b2 = np.random.random(size=(outputSize)) - 0.5
        self.loss_history = []

    def train(self, X, y, learning_rate=1e-2,reg = 1e-4,decay_rate=0.95,
              n_iters=5000, batch_size=200, verbose=1):
        lr = learning_rate
        for i in xrange(n_iters):
            # adding dense layer1
            ids = np.random.choice(X.shape[0], batch_size)
            l1out, l1cache = layers.dense_forward(X[ids], self.W1, self.b1)
            # adding relu layer2
            l2out, l2cache = layers.relu_forward(l1out)
            # adding dense layer3
            l3out, l3cache = layers.dense_forward(l2out, self.W2, self.b2)
            # adding softmax layer
            loss, l4cache = layers.softmax_loss_forward(l3out, y[ids])
            loss = loss + reg*(np.sum(self.W1**2) + np.sum(self.W2**2))
            self.loss_history.append(loss)
            if verbose and i % 500 == 0:
            	lr = lr*0.95
                print "Iteration %d, loss = %f" % (i, loss)
            dlayer4 = 1.0
            dlayer3 = layers.softmax_loss_backward(dlayer4, l4cache)
            dlayer2, dW2, db2 = layers.dense_backward(dlayer3, l3cache)
            dlayer1 = layers.relu_backward(dlayer2, l2cache)
            _, dW1, db1 = layers.dense_backward(dlayer1, l1cache)
            self.W1 = self.W1 - lr * dW1 + 0.5*reg*dW1
            self.b1 = self.b1 - lr * db1
            self.W2 = self.W2 - lr * dW2 + 0.5*reg*dW2
            self.b2 = self.b2 - lr * db2

    def predict(self, X):
        l1out, _ = layers.dense_forward(X, self.W1, self.b1)
        l2out, _ = layers.relu_forward(l1out)
       	l3out, _ = layers.dense_forward(l2out, self.W2, self.b2)
       	return np.argmax(l3out, axis=1)

    def accuracy(self, X, y):
       	return np.mean(self.predict(X) == y)

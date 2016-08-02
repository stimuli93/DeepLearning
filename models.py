import layers
import rnn_layers
import numpy as np
import initializations
import optimizers


class MLP(object):

    """
    2 hidden layer neural network with softmax classifier
    """
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize, hiddenLayer='relu'):
        self.inputSize = inputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2
        self.outputSize = outputSize
        self.hiddenLayer = hiddenLayer
        self.loss_history = []
        self.gradientLayer1 = []
        self.gradientLayer2 = []
        self.params = {}

        self.W1 = initializations.xavier_init((inputSize, hiddenSize1), hiddenLayer=hiddenLayer)
        self.b1 = initializations.uniform_init((hiddenSize1,))
        self.W2 = initializations.xavier_init((hiddenSize1, hiddenSize2), hiddenLayer=hiddenLayer)
        self.b2 = initializations.uniform_init((hiddenSize2,))
        self.W3 = initializations.xavier_init((hiddenSize2, outputSize), hiddenLayer=hiddenLayer)
        self.b3 = initializations.uniform_init((outputSize,))

    def train(self, X, y, X_val=None, y_val=None, learning_rate=1e-2, reg = 1e-4, decay_rate=0.95, opt='sgd',
              n_iters=5000, batch_size=200, verbose=1):
        lr = learning_rate
        for i in xrange(n_iters):
            # adding dense layer1
            ids = np.random.choice(X.shape[0], batch_size)
            l1out, l1cache = layers.dense_forward(X[ids], self.W1, self.b1)
            # adding non-linearity layer2
            l2out, l2cache = layers.non_linearity_forward(l1out,self.hiddenLayer)
            # adding dense layer3
            l3out, l3cache = layers.dense_forward(l2out, self.W2, self.b2)
            # adding non-linearity layer4
            l4out,l4cache = layers.non_linearity_forward(l3out, self.hiddenLayer)
            # adding dense layer5
            l5out,l5cache = layers.dense_forward(l4out, self.W3, self.b3)
            # adding softmax layer
            loss, l6cache = layers.softmax_loss_forward(l5out, y[ids])
            loss = loss + 0.5*reg*(np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
            self.loss_history.append(loss)
            if verbose and i % 500 == 0:
                lr *= decay_rate
                print "Iteration %d, loss = %f" % (i, loss)
                if X_val is not None and y_val is not None:
                    print "Validation Accuracy :%f" % (self.accuracy(X_val, y_val))

            dlayer6 = 1.0
            dlayer5 = layers.softmax_loss_backward(dlayer6, l6cache)
            dlayer4, dW3, db3 = layers.dense_backward(dlayer5, l5cache)
            dlayer3 = layers.non_linearity_backward(dlayer4, l4cache, self.hiddenLayer)
            dlayer2, dW2, db2 = layers.dense_backward(dlayer3, l3cache)
            dlayer1 = layers.non_linearity_backward(dlayer2, l2cache, self.hiddenLayer)
            _, dW1, db1 = layers.dense_backward(dlayer1, l1cache)

            self.gradientLayer1.append(np.mean(np.abs(dlayer1)))
            self.gradientLayer2.append(np.mean(np.abs(dlayer3)))

            self.params, self.W1 = optimizers.optimize(self.params, self.W1, dW1, lr=lr, name='W1', opt=opt)
            self.params, self.b1 = optimizers.optimize(self.params, self.b1, db1, lr=lr, name='b1', opt=opt)
            self.params, self.W2 = optimizers.optimize(self.params, self.W2, dW2, lr=lr, name='W2', opt=opt)
            self.params, self.b2 = optimizers.optimize(self.params, self.b2, db2, lr=lr, name='b2', opt=opt)
            self.params, self.W3 = optimizers.optimize(self.params, self.W3, dW3, lr=lr, name='W3', opt=opt)
            self.params, self.b3 = optimizers.optimize(self.params, self.b3, db3, lr=lr, name='b3', opt=opt)

            # gradients due to regularization
            self.W1 += reg * dW1
            self.W2 += reg * dW2
            self.W3 += reg * dW3

    def predict(self, X):

        # return the highest value for each row after a forward pass
        l1out, _ = layers.dense_forward(X, self.W1, self.b1)
        l2out, _ = layers.non_linearity_forward(l1out, self.hiddenLayer)
        l3out, _ = layers.dense_forward(l2out, self.W2, self.b2)
        l4out, _ = layers.non_linearity_forward(l3out,self.hiddenLayer)
        l5out, _ = layers.dense_forward(l4out, self.W3, self.b3)
        return np.argmax(l5out, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


class RNN_SA(object):
    """
    Recurrent Neural Network Specifically for the task of Sentiment Analysis
    """
    def __init__(self, input_dim, hidden_dim, output_dim, non_linearity='tanh'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.non_liniearity = non_linearity
        self.Wx = initializations.xavier_init((input_dim, hidden_dim), hiddenLayer=non_linearity)
        self.Wh = initializations.xavier_init((hidden_dim, hidden_dim), hiddenLayer=non_linearity)
        self.W1 = initializations.xavier_init((hidden_dim, output_dim), hiddenLayer='relu')
        self.b1 = initializations.uniform_init((output_dim,))
        self.b = initializations.uniform_init((hidden_dim,))
        self.loss_history = []
        self.params = {}

    def train(self, X, y, learning_rate=1e-2, opt='sgd', n_iters=5000, batch_size=200, verbose=1):
        lr = learning_rate
        N, T, D = X.shape
        for i in xrange(n_iters):
            ids = np.random.choice(X.shape[0], batch_size)
            h0 = np.zeros((batch_size, self.hidden_dim))
            layer1, l1cache = rnn_layers.rnn_forward(X[ids], h0, self.Wx, self.Wh, self.b, self.non_liniearity)
            final_layer = (layer1[:, T-1, :])
            layer2, l2cache = layers.dense_forward(final_layer, self.W1, self.b1)
            loss, l3cache = layers.softmax_loss_forward(layer2, y[ids])
            self.loss_history.append(loss)

            if verbose == 1 and i % 500 == 0:
                print 'Iteration %d: loss %g' % (i, loss)

            dlayer3 = 1.0
            dlayer2 = layers.softmax_loss_backward(dlayer3, l3cache)
            dlayer1, dW1, db1 = layers.dense_backward(dlayer2, l2cache)
            dh = np.zeros((batch_size, T, self.hidden_dim))
            dh[:, T-1, :] = dlayer1
            _, _, dWx, dWh, db = rnn_layers.rnn_backward(dh, l1cache)

            self.params, self.Wx = optimizers.optimize(self.params, self.Wx, dWx, lr=lr, name='Wx', opt=opt)
            self.params, self.Wh = optimizers.optimize(self.params, self.Wh, dWh, lr=lr, name='Wh', opt=opt)
            self.params, self.b = optimizers.optimize(self.params, self.b, db, lr=lr, name='b', opt=opt)
            self.params, self.W1 = optimizers.optimize(self.params, self.W1, dW1, lr=lr, name='W1', opt=opt)
            self.params, self.b1 = optimizers.optimize(self.params, self.b1, db1, lr=lr, name='b1', opt=opt)

    def predict(self, X):
        N, T, D = X.shape
        h0 = np.zeros((N, self.hidden_dim))
        layer1, l1cache = rnn_layers.rnn_forward(X, h0, self.Wx, self.Wh, self.b, self.non_liniearity)
        final_layer = (layer1[:, T - 1, :])
        layer2, _ = layers.dense_forward(final_layer, self.W1, self.b1)
        return np.argmax(final_layer, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

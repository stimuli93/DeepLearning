import layers
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
            l5out,l5cache = layers.dense_forward(l4out,self.W3, self.b3)
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

import layers
import numpy as np
import initializations
import optimizers


class SimpleAutoEncoder(object):
    """
    Simple auto-encoder with binary cross-entropy loss
    """
    def __init__(self, inputSize, hiddenSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.loss_history = []
        self.W1 = initializations.xavier_init(shape=(inputSize, hiddenSize), hiddenLayer='relu')
        self.b1 = initializations.uniform_init(shape=(hiddenSize,))

        self.W2 = initializations.xavier_init(shape=(hiddenSize, inputSize), hiddenLayer='sigmoid')
        self.b2 = initializations.uniform_init(shape=(inputSize,))
        self.params = {}
        self.reg = 1e-5

    def train(self, X, y, learning_rate=1e-3, reg=1e-4, decay_rate=1.00, opt='sgd', n_iters=1000,
              batch_size=200, verbose=True):
        lr = learning_rate
        self.reg = reg
        for i in range(n_iters):
            ids = np.random.choice(X.shape[0], batch_size)
            layer1, l1cache = layers.dense_forward(X[ids], self.W1, self.b1)
            layer2, l2cache = layers.non_linearity_forward(layer1, hiddenLayer='relu')
            layer3, l3cache = layers.dense_forward(layer2, self.W2, self.b2)
            layer4, l4cache = layers.non_linearity_forward(layer3, hiddenLayer='sigmoid')
            loss, l5cache = layers.binary_cross_entropy_loss_forward(layer4, y[ids])

            dlayer5 = 1.0
            dlayer4 = layers.binary_cross_entropy_loss_backward(dlayer5, l5cache)
            dlayer3 = layers.non_linearity_backward(dlayer4, l4cache, hiddenLayer='sigmoid')
            dlayer2, dW2, db2 = layers.dense_backward(dlayer3, l3cache)
            dlayer1 = layers.non_linearity_backward(dlayer2, l2cache, hiddenLayer='relu')
            _, dW1, db1 = layers.dense_backward(dlayer1, l1cache)

            loss += 0.5 * reg * (np.sum(self.W1*self.W1) + np.sum(self.W2*self.W2))

            if i % 500 == 0:
                lr *= decay_rate
                if verbose:
                    print "Iteration %d, loss = %g" % (i, loss)

            self.params, self.W1 = optimizers.optimize(self.params, self.W1, dW1, lr=lr, name='W1', opt=opt)
            self.params, self.b1 = optimizers.optimize(self.params, self.b1, db1, lr=lr, name='b1', opt=opt)
            self.params, self.W2 = optimizers.optimize(self.params, self.W2, dW2, lr=lr, name='W2', opt=opt)
            self.params, self.b2 = optimizers.optimize(self.params, self.b2, db2, lr=lr, name='b2', opt=opt)

            dW2 += 0.5 * reg * self.W2
            dW1 += 0.5 * reg * self.W1
            self.loss_history.append(loss)

    def predict(self, X):
        l1, _ = layers.dense_forward(X, self.W1, self.b1)
        l2, _ = layers.non_linearity_forward(l1, hiddenLayer='relu')
        l3, _ = layers.dense_forward(l2, self.W2, self.b2)
        l4, _ = layers.non_linearity_forward(l3, hiddenLayer='sigmoid')
        return l4

    def getloss(self, X, y):
        layer1, _ = layers.dense_forward(X, self.W1, self.b1)
        layer2, _ = layers.non_linearity_forward(layer1, hiddenLayer='relu')
        layer3, _ = layers.dense_forward(layer2, self.W2, self.b2)
        layer4, _ = layers.non_linearity_forward(layer3, hiddenLayer='sigmoid')
        loss, _ = layers.binary_cross_entropy_loss_forward(layer4, y)
        loss += 0.5 * self.reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return loss

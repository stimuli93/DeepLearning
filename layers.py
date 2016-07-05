import numpy as np


def dense_forward(x, w, b):
    """
    :param x: numpy array of shape (N,D) where N is number of training examples each having D features
    :param w: weights of type numpy array having shape (D,H) where H is the number of output units
    :param b: bias of shape (H,) representing bias for output layer
    :return:
    out = numpy array of shape (N,H) representing the output layer
    cache = tuple of input parameters used for backpropagation
    """
    out = np.dot(x, w) + b
    cache = (x, w, b)
    return out, cache


def dense_backward(dout, cache):
    """
    :param dout: gradient of the output layers having shape (N,H)
    :param cache: tuple of input layer, weights & bias
    :return:
    dx = numpy array of shape (N,D) representing gradients of input layer
    dw = numpy array of shape (D,H) representing gradients of weights
    db = numpy array of shape (H,) representing gradients of bias
    """
    x, w, b = cache
    dx = np.dot(dout, w.T)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def sigmoid_forward(x):
    """
    :param x: numpy array of shape (N,D) representing input layer
    :return:
    out = numpy array of shape (N,D) representing output of sigmoid layer
    cache = storing x for backpropagation
    """
    cache = x
    out = 1/(1 + np.exp(-x))
    return out, cache


def sigmoid_backward(dout, cache):
    """
    :param dout: numpy array of shape (N,D) representing gradients of output layer
    :param cache: numpy array of shape (N,D) representing input layer for backpropagation
    :return:
    dx = numpy array of shape (N,D) representing gradients of input layer
    """
    x = cache
    out = 1/(1 + np.exp(-x))
    dx = dout*(out*(1-out))
    return dx


def tanh_forward(x):
    """
    :param x: numpy array of shape (N,D) representing input layer
    :return:
    out = numpy array of shape (N,D) representing output of tanh layer
    cache = storing x for backpropagation
    """
    cache = x
    out = np.tanh(x)
    return out, cache


def tanh_backward(dout, cache):
    """
    :param dout: numpy array of shape (N,D) representing gradients of output layer
    :param cache: numpy array of shape (N,D) representing input layer for backpropagation
    :return:
    dx = numpy array of shape (N,D) representing gradients of input layer
    """
    x = cache
    out = np.tanh(x)
    dx = dout*(1-out**2)
    return dx


def relu_forward(x):
    """
    :param x: numpy array of shape (N,D) representing input layer
    :return:
    out = numpy array of shape (N,D) representing output of relu layer
    cache = storing x for backpropagation
    """
    out = x*(x > 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    :param dout: numpy array of shape (N,D) representing gradients of output layer
    :param cache: numpy array of shape (N,D) representing input layer for backpropagation
    :return:
    dx = numpy array of shape (N,D) representing gradients of input layer
    """
    x = cache
    dx = dout*(x > 0)
    return dx


def non_linearity_forward(x, hiddenLayer):
    if hiddenLayer == 'tanh':
        return tanh_forward(x)
    if hiddenLayer == 'sigmoid':
        return sigmoid_forward(x)
    return relu_forward(x)      


def non_linearity_backward(dout, cache, hiddenLayer):
    if hiddenLayer == 'tanh':
        return tanh_backward(dout, cache)
    if hiddenLayer == 'sigmoid':
        return sigmoid_backward(dout, cache)
    return relu_backward(dout, cache)


def softmax_loss_forward(x, y):
    """
    :param x: numpy array of shape (N,C) representing input layer
    :param y: numpy array of shape (N,) representing labels of each training example
    :return:
    loss = softmax loss
    cache = tuple of (x,y) used for backpropagation
    """
    cache = (x, y)
    x_exp = np.exp((x.T - np.max(x, axis=1)).T)
    probabilities = (x_exp.T / np.sum(x_exp, axis=1)).T
    N = x.shape[0]
    loss = -np.mean(np.log(probabilities[xrange(N), y]))
    return loss, cache


def softmax_loss_backward(dout, cache):
    """
    :param dout: Gradient coming from 1 layer deeper
    :param cache: (x,y) used for backpropagation
    :return:
    dx = numpy array of shape (N,C) representing gradients of input layer
    """
    x, y = cache
    x_exp = np.exp((x.T - np.max(x, axis=1)).T)
    probabilities = (x_exp.T / np.sum(x_exp, axis=1)).T
    N = x.shape[0]
    dx = probabilities
    tmp = np.zeros(x.shape)
    tmp[xrange(N), y] = 1.0
    dx = (dx - tmp)/N
    dx = dx * dout
    return dx

import numpy as np

class SGD(object):
    """
    Stochastic Gradient Descent optimizer requires only learning_rate parameter for updating weights
    The params dictionary is not modified by this optimizer
    """
    def __init__(self, lr=1e-3):
        self.lr = lr

    def update(self, w, params, grad, name):
        w -= self.lr * grad
        return params, w


class Momentum(object):

    def __init__(self, lr=1e-3):
        self.lr = lr

    def update(self, w, params, grad, name):
        mu = params.get('mu', 0.75)
        params['mu'] = mu
        v = np.zeros(w.shape)
        v = params.get(name + 'v', v)
        v = mu * v - self.lr * grad
        params[name + 'v'] = v
        w += v
        return params, w


class Adagrad(object):

    def __init__(self, lr=1e-3):
        self.lr = lr

    def update(self, w, params, grad, name):
        cache = np.zeros(grad.shape)
        cache = params.get(name + 'cache', cache)
        cache += (grad*grad)
        params[name + 'cache'] = cache
        w -= ((self.lr * grad)/(np.sqrt(cache) + 1e-7))
        return params, w


class Rmsprop(object):

    def __init__(self, lr=1e-3):
        self.lr = lr

    def update(self, w, params, grad, name):
        cache = np.zeros(grad.shape)
        cache = params.get(name + 'cache', cache)
        decay_rate = params.get('decay_rate',0.99)
        cache = decay_rate*cache + (1-decay_rate)*(grad * grad)
        params[name + 'cache'] = cache
        params['decay_rate'] = decay_rate
        w -= ((self.lr * grad) / (np.sqrt(cache) + 1e-7))
        return params, w


class Adam(object):

    def __init__(self, lr=1e-3):
        self.lr = lr

    def update(self, w, params, grad, name):
        v = np.zeros(grad.shape)
        cache = np.zeros(grad.shape)
        v = params.get(name + 'v', v)
        cache = params.get(name + 'cache', cache)
        beta1 = params.get('beta1',0.9)
        beta2 = params.get('beta2',0.99)

        v = beta1*v + (1-beta1)*grad
        cache = beta2*cache + (1-beta2)*(grad * grad)
        params[name + 'cache'] = cache
        params[name + 'v'] = v
        w -= ((self.lr * v) / (np.sqrt(cache) + 1e-7))
        return params, w


def optimize(params, w, grad, name, lr=1e-3, opt='sgd'):
    """
    :param params: a dictionary which contains parameters like beta, momentum, etc used for weight updation
    :param w: a numpy array of weights which are to be updated
    :param grad: a numpy array of same dimension as w representing gradients of w
    :param name: value like W1/b2 etc used for in formation of key of params dictionary
    :param lr: learning rate
    :param opt: the optimizer to be used
    :return:
    (params, w) a tuple representing the updated dictionary and weight for the model
    """
    if opt == 'momentum':
        momentum = Momentum(lr)
        return momentum.update(w, params=params, grad=grad, name=name)

    if opt == 'adagrad':
        adagrad = Adagrad(lr)
        return adagrad.update(w, params=params, grad=grad, name=name)

    if opt == 'rmsprop':
        rmsprop = Rmsprop(lr)
        return rmsprop.update(w, params=params, grad=grad, name=name)

    if opt == 'adam':
        adam = Adam(lr)
        return adam.update(w, params=params, grad=grad, name=name)

    sgd = SGD(lr)
    return sgd.update(w, params=params, grad=grad, name=name)

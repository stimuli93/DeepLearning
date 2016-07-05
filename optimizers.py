

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
    if opt == 'sgd':
        sgd = SGD(lr)
        return sgd.update(w, params=params, grad=grad, name=name)

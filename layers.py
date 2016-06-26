import numpy as np

class Layer(object):
    
    def dense_forward(x,w,b):
        """
        Inputs
        x = numpy array of shape (N,D) where N is number of training examples
        each having D features
        w = weights of type numpy array having shape (D,H) where H is the
        number of output units
        b = bias of shape (H,) representing bias for output layer

        Return
        out = numpy array of shape (N,H) representing the output layer
        cache = tuple of input parameters used for backpropagation

        """
        out = np.dot(x,w) + b
        cache = (x,w,b)
        return out,cache

    def dense_backward(dout,cache):
        """
        Inputs
        dout = gradient of the output layers having shape (N,H)
        cache = tuple of input layer, weights & bias

        Return
        dx = numpy array of shape (N,D) representing gradients of input layer
        dw = numpy array of shape (D,H) representing gradients of weights
        db = numpy array of shape (H,) representing gradients of bias
        """
        x,w,b = cache
        dx = np.dot(dout,w.T)
        dw = np.dot(x.T,dout)
        db = np.sum(dout,axis=0)
        return dx,dw,db

    def relu_forward(x):
        """
        Inputs
        x = numpy array of shape (N,D) representing input layer

        Return
        out = numpy array of shape (N,D) representing output of relu layer
        cache = storing x for backpropagation
        """
        out = x*(x>0)
        cache = x
        return out,cache

    def relu_backward(dout,cache):
        x = cache
        dx = dout*(x>0)
        return dx




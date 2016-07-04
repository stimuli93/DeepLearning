import numpy as np

def uniform_init(shape):
    """
    Inputs
    shape = tuple of integers representing shape of matrix

    Return
    out = numpy array of given shape
    """
    if (len(shape) > 1):
        out = np.random.randn(shape[0],shape[1])
    else:
        out = np.random.randn(shape[0])
	
    return out

def xavier_init(shape, hiddenLayer='relu'):
    """
    Inputs
    shape = tuple of integers representing shape of matrix

    Return
    out = numpy array of given shape
    """
    if (len(shape) > 1):
        out = np.random.randn(shape[0],shape[1])
    else:
        out = np.random.randn(shape[0])
   
    if hiddenLayer == 'relu':
        out = out / np.sqrt(shape[0]/2.0)
    else:
        out = out / np.sqrt(shape[0])
    return out


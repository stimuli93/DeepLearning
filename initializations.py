import numpy as np


def uniform_init(shape):
    """

    :param shape: tuple of integers representing shape of matrix
    :return:
    out: numpy array of given shape
    """
    if len(shape) > 1:
        out = np.random.randn(shape[0], shape[1])
    else:
        out = np.random.randn(shape[0])

    return out


def xavier_init(shape, hiddenLayer='relu'):
    """
    :param shape: tuple of integers representing shape of matrix
    :param hiddenLayer: non-linearity to be used
    :return:
    out: numpy array of given shape
    """
    if len(shape) > 1:
        out = np.random.randn(shape[0], shape[1])
    else:
        out = np.random.randn(shape[0])
   
    if hiddenLayer == 'relu':
        out = out / np.sqrt(shape[0]/2.0)
    else:
        out = out / np.sqrt(shape[0])
    return out


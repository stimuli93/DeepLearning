import numpy as np
import math


def uniform_init(shape, low=-1.0, high=1.0):
    """
    :param shape: tuple of integers representing shape of matrix
    :param low:
    :param high:
    :return:
    out: numpy array of given shape
    """
    return np.random.uniform(low, high, size=shape)


def xavier_init(shape, uniform=True):
    """
    :param shape: tuple of integers representing input & output dimensions
    :param uniform:
    :return:
    numpy array of given shape
    """
    if uniform:
        init_range = math.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-init_range, init_range, size=shape)
    else:
        stddev = math.sqrt(3.0 / (shape[0] + shape[1]))
        return np.random.normal(scale=stddev,size=shape)


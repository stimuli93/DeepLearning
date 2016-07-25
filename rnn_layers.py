import numpy as np
import layers


def rnn_step_forward(x, prev_h, Wx, Wh, b, non_liniearity='tanh'):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses the specified
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    :param x: Input data for this timestep, of shape (N, D)
    :param prev_h: Hidden state from previous timestep, of shape (N, H)
    :param Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    :param Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    :param b: Biases of shape (H,)
    :param non_liniearity: relu/sigmoid or tanh non-linearity to be used
    :return:
    :next_h: Next hidden state, of shape (N, H)
    :cache: Tuple of values needed for the backward pass.
    """
    tmp = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    next_h, _ = layers.non_linearity_forward(tmp, hiddenLayer=non_liniearity)
    cache = (x, prev_h, Wx, Wh, b, non_liniearity)
    return next_h, cache


def rnn_step_backward(dout, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    :param dnext_h: Gradient of loss with respect to next hidden state
    :param cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    x, prev_h, Wx, Wh, b, non_linearity = cache
    tmp = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    d_tmp = layers.non_linearity_backward(dout, tmp, hiddenLayer=non_linearity)
    db = np.sum(d_tmp, axis=0)
    dx = np.dot(d_tmp, Wx.T)
    dprev_h = np.dot(d_tmp, Wh.T)
    dWx = np.dot(x.T, d_tmp)
    dWh = np.dot(prev_h.T, d_tmp)
    return dx, dprev_h, dWx, dWh, db

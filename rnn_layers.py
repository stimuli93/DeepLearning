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


def rnn_forward(x, h0, Wx, Wh, b, non_linearity='tanh'):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    N, T, D = x.shape
    H = h0.shape[1]
    cache = []
    h = np.zeros((N, T, H))
    for i in xrange(T):
        xi = x[:, i, :]
        h0, cache_i = rnn_step_forward(xi, prev_h=h0, Wx=Wx, Wh=Wh, b=b, non_liniearity=non_linearity)
        h[:, i, :] = h0
        cache.append(cache_i)
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    N, T, H = dh.shape
    tx, dh0, dWx, dWh, db, _ = cache[0]
    dx = np.zeros((N, T, tx.shape[1]))
    dh0 = np.zeros(dh0.shape)
    dWx = np.zeros(dWx.shape)
    dWh = np.zeros(dWh.shape)
    db = np.zeros(db.shape)
    for i in reversed(xrange(T)):
        cache_i = cache[i]
        dh_i = dh[:, i, :]
        dxt, dh0, dWxt, dWht, dbt = rnn_step_backward(dh_i, cache_i)
        dx[:, j, :] = dxt
        dWx += dWxt
        dWh += dWht
        db += dbt

    return dx, dh0, dWx, dWh, db


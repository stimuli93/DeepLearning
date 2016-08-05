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
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N,T,H = dh.shape
    tx, dh0, dWx, dWh, db, _ = cache[0]
    dx = np.zeros((N, T, tx.shape[1]))
    dh0 = np.zeros(dh0.shape)
    dWx = np.zeros(dWx.shape)
    dWh = np.zeros(dWh.shape)
    db = np.zeros(db.shape)

    for i in xrange(T):
        j = T-i-1
        cache_t = cache[j]
        dht = dh[:,j,:] + dh0
        dxt, dh0, dWxt, dWht, dbt = rnn_step_backward(dht, cache_t)
        dx[:, j, :] = dxt
        dWx += dWxt
        dWh += dWht
        db += dbt
    return dx, dh0, dWx, dWh, db


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    N, H = prev_h.shape
    tmp = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    tmp_i = tmp[:, :H]
    tmp_f = tmp[:, H:2*H]
    tmp_o = tmp[:, 2*H:3*H]
    tmp_g = tmp[:, 3*H:]

    next_c = tmp_f*prev_c + tmp_i*tmp_g
    next_h = tmp_o * np.tanh(next_c)
    cache = (x, prev_h, prev_c, Wx, Wh, b)
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    x, prev_h, prev_c, Wx, Wh, b = cache
    N, H = prev_h.shape
    tmp = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    tmp_i = tmp[:, :H]
    tmp_f = tmp[:, H:2 * H]
    tmp_o = tmp[:, 2 * H:3 * H]
    tmp_g = tmp[:, 3 * H:]
    next_c = tmp_f * prev_c + tmp_i * tmp_g

    dnext_c += dnext_h*tmp_o*(1-(np.tanh(next_c)**2))
    dtmp_o = dnext_h*np.tanh(next_c)
    dtmp_f = dnext_c*prev_c
    dtmp_i = dnext_c*tmp_g
    dtmp_g = dnext_c*tmp_i
    dprev_c = dnext_c*tmp_f

    dtmp = np.hstack((dtmp_i, dtmp_f, dtmp_o, dtmp_g))
    db = np.sum(dtmp, axis=0)
    dx = np.dot(dtmp, Wx.T)
    dprev_h = np.dot(dtmp, Wh.T)
    dWx = np.dot(x.T, dtmp)
    dWh = np.dot(prev_h.T, dtmp)
    return dx, dprev_h, dprev_c, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    cache = (x, W)
    out = W[x, :]
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros(W.shape)
    np.add.at(dW, x, dout)

    return dW


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    N, T, D = x.shape
    H = h0.shape[1]
    h = np.zeros((N, T, H))
    c0 = np.zeros(h0.shape)
    cache = []
    for i in xrange(T):
        xi = x[:, i, :]
        h0, c0, cache_i = lstm_step_forward(xi, h0, c0, Wx, Wh, b)
        h[:, i, :] = h0
        cache.append(cache_i)
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    N, T, H = dh.shape
    tx, dc0, dh0, dWx, dWh, db = cache[0]
    D = tx.shape[1]
    dc0 = np.zeros(dc0.shape)
    dh0 = np.zeros(dh0.shape)
    dWx = np.zeros(dWx.shape)
    dWh = np.zeros(dWh.shape)
    db = np.zeros(db.shape)
    dx = np.zeros((N, T, D))

    for i in xrange(T):
        j = T-i-1
        cache_j = cache[j]
        dh0 += dh[:, j, :]
        dx_j, dh0, dc0, dWx_j, dWh_j, db_j = lstm_step_backward(dh0, dc0, cache_j)
        dx[:, j, :] = dx_j
        dWx += dWx_j
        dWh += dWh_j
        db += db_j

    return dx, dh0, dWx, dWh, db


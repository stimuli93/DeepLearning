import numpy as np
import layers
import rnn_layers

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print ix, grad[ix]
        it.iternext() # step to next dimension

    return grad


def test_denselayer():
    x = np.random.randn(10, 6)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: layers.dense_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: layers.dense_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: layers.dense_forward(x, w, b)[0], b, dout)

    _, cache = layers.dense_forward(x, w, b)
    dx, dw, db = layers.dense_backward(dout, cache)

    # The error should be around 1e-10
    print 'Testing dense layers:'
    print 'dx error: ', rel_error(dx_num, dx)
    print 'dw error: ', rel_error(dw_num, dw)
    print 'db error: ', rel_error(db_num, db)


def test_relulayer():
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: layers.relu_forward(x)[0], x, dout)
    _, cache = layers.relu_forward(x)
    dx = layers.relu_backward(dout, cache)

    # The error should be around 1e-12
    print 'Testing relu layers:'
    print 'dx error: ', rel_error(dx_num, dx)


def test_tanhlayer():
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: layers.tanh_forward(x)[0], x, dout)
    _, cache = layers.tanh_forward(x)
    dx = layers.tanh_backward(dout, cache)

    # The error should be around 1e-12
    print 'Testing tanh layers:'
    print 'dx error: ', rel_error(dx_num, dx)


def test_sigmoidlayer():
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: layers.sigmoid_forward(x)[0], x, dout)
    _, cache = layers.sigmoid_forward(x)
    dx = layers.sigmoid_backward(dout, cache)

    # The error should be around 1e-12
    print 'Testing sigmoid layers:'
    print 'dx error: ', rel_error(dx_num, dx)


def test_softmax_loss():
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)
    cache = (x, y)
    dout = 1.0

    dx_num = eval_numerical_gradient(lambda x: layers.softmax_loss_forward(x, y)[0], x, verbose=False)
    dx = layers.softmax_loss_backward(dout, cache)

    # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
    print 'Testing softmax_loss:'
    print 'dx error: ', rel_error(dx_num, dx)


def test_binary_cross_entropy_loss():
    x = np.random.rand(10, 10)
    y = np.random.rand(*x.shape) < 0.5
    cache = (x, y)
    dout = 1.0

    dx_num = eval_numerical_gradient_array(lambda x: layers.binary_cross_entropy_loss_forward(x, y)[0], x, dout)
    dx = layers.binary_cross_entropy_loss_backward(dout, cache)

    # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
    print 'Testing binary cross-entropy loss:'
    print 'dx error: ', rel_error(dx_num, dx)


def test_rnn_step_layer():
    N, D, H = 4, 5, 6
    x = np.random.randn(N, D)
    h = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    out, cache = rnn_layers.rnn_step_forward(x, h, Wx, Wh, b)

    dnext_h = np.random.randn(*out.shape)

    fx = lambda x: rnn_layers.rnn_step_forward(x, h, Wx, Wh, b)[0]
    fh = lambda prev_h: rnn_layers.rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWx = lambda Wx: rnn_layers.rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWh = lambda Wh: rnn_layers.rnn_step_forward(x, h, Wx, Wh, b)[0]
    fb = lambda b: rnn_layers.rnn_step_forward(x, h, Wx, Wh, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
    dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
    db_num = eval_numerical_gradient_array(fb, b, dnext_h)

    dx, dprev_h, dWx, dWh, db = rnn_layers.rnn_step_backward(dnext_h, cache)

    print 'Testing rnn_step layers'
    print 'dx error: ', rel_error(dx_num, dx)
    print 'dprev_h error: ', rel_error(dprev_h_num, dprev_h)
    print 'dWx error: ', rel_error(dWx_num, dWx)
    print 'dWh error: ', rel_error(dWh_num, dWh)
    print 'db error: ', rel_error(db_num, db)


if __name__ == '__main__':
    test_denselayer()
    test_relulayer()
    test_tanhlayer()
    test_sigmoidlayer()
    test_softmax_loss()
    test_binary_cross_entropy_loss()
    test_rnn_step_layer()

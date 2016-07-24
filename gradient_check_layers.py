import numpy as np
import layers

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



if __name__ == '__main__':
    test_denselayer()
    test_relulayer()
    test_tanhlayer()
    test_sigmoidlayer()
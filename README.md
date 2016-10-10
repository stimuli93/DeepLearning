## Fundamentals of DeepLearning

*layers.py* consists of forward & backward pass for the following layers:
* dense layer (fully-connected layer)
* sigmoid layer
* tanh layer
* relu layer
* dropout layer
* softmax_loss layer
* svm_loss layer

*rnn_layers.py* consists of forward & backward pass of:
* vanilla rnn
* lstm
* embedding layer

*gradient_check_layers.py* consists of gradient checking of the above mentioned layers

*optimizers.py* consists of implementation of following optimizers:
* SGD
* Momentum
* Adagrad
* Rmsprop
* Adam

*initializations.py* consists of following initializations schemes:
* uniform initialization
* xavier initialization

*Activation Analysis.ipynb* is an iPython notebook which performs Analysis of Gradient at hidden layers of Relu,
Sigmoid and Tanh Activations to study the vanishing gradient problem.

Summary: 
* For tanh & relu non-linearities the mean of absolute gradient is similar on both hidden layers but for the sigmoid non-linearity the avg. of absolute gradient is significantly higher at the second layer compared to the first i.e vanishing gradient problem is much more evident in sigmoid non-linearity than on tanh or relu.
* Mean absolute gradient is highest in sigmoid layer among different non-linearities.
* Mean absolute gradient is higher in case of a sparse neural network when compared to its denser counterpart.

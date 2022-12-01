"""
===================
ANN in numpy
===================
"""

# %%imports
# In this tutorial, we will build and train a multi layer perceptron from scratch
# in numpy. We will implement the forward pass, backward pass, weight update so
# that we can get the idea what actually happens under the hood when we train
# neural networks for a particular data. The main purpose is to understand the
# forward and backward propagation and its implementation.
#
# .. code-block:: python
#
#    import numpy as np
#

# %%
# data preparation
# ------------------
# The purpose of data preparation step in a supervised learning task is to divide our
# data into input/output pairs. The number of input/output paris should be equal. We
# can call one pair of input and its corresponding output as ``example``. We feed
# many such examples to the neural network to make it learn the relationship between
# inputs and outputs.
#
# .. code-block:: python
#
#     from ai4water.datasets import busan_beach
#     data = busan_beach()
#     print(data.shape)
#     # 1446, 14
#

# %%
# splitting
# -----------
# The length of the data is above 1400. However, not the target column consists of many missing
# values. This will reduce the number of examples that we will finally have at our disposal.
# Furthermore, we will divide the total number of examples into training and validation sets.
# We will use 70% of the examples for training and 30% for the validation. The splitting is
# performed randomly. The value of seed 2809 is for reproducibility purpose.
#
# .. code-block:: python
#
#     from ai4water.preprocessing import DataSet
#     dataset = DataSet(data, val_fraction=0.0, seed=2809)
#     X_train, y_train = dataset.training_data()
#     X_val, y_val = dataset.test_data()
#

# %%
# batch generation
#
# .. code-block:: python
#
#    def batch_generator(X, Y, size=32):
#
#        for ii in range(0, len(X), size):
#            X_batch, y_batch = X[ii:ii + size], Y[ii:ii + size]
#
#            yield X_batch, y_batch
#
# The purpose of ``batch_generator`` function is to divided out data (x,y) into batches.
# The size of each batch is determined by ``size`` argument.

# %%
# hyperparameters
# Next we define hyperparameters of out feed forward neural network or multi-layer perceptron.
# The hyeparameters are those parameters which determine how the parameter are going to be
# estimated. These ``parameters`` here mean weights and biases whose values are calibrated/optimized
# due the training process.
#
# .. code-block:: python
#
#    lr = 0.01
#    epochs = 1000
#    l1_neurons = 10
#    l2_neurons = 5
#    l3_neurons = 1
#
# ``lr`` is the learning rate. It determines the jump in the values of weights and biases
# which we make at each parameter update step. The parameter update step can be performed
# either after feeding the whole data to the neural network or feeding a single example
# to the network or a batch of examples to the network. In this example, we will update
# the parameters after each batch. The hyperparameter ``epochs`` determine, how many
# times we want to show our whole data to the neural network. The values of neurons determine
# the size of learnable parameters i.e., weights and biases in each layer. The larger
# the neurons, the bigger is the size of weights and biases matrices, the larger
# is the learning capacity of the network, the higher is the computation cost. The number
# of neurons in the last layer must match the number of target/output variables. In our
# case we have just one target variable.

# %%
# weights and biases
# -----------------------
# Next, we initialize our weights and biases with random numbers. We will have three
# layers, 2 hidden and 1 output. Each of these layers will have two learnable parameters
# i.e. weights and biases. The size of the weights and biases in a layer depends upon
# the size of inputs that it is receiving and a user defined parameter i.e. ``neurons`` which
# is also calle ``units``.
#
# .. code-block:: python
#
#    from numpy.random import default_rng
#
#    rng_l1 = default_rng(313)
#    rng_l2 = default_rng(313)
#    rng_l3 = default_rng(313)
#    w1 = rng_l1.standard_normal((dataset.num_ins, l1_neurons))
#    b1 = rng_l1.standard_normal((1, l1_neurons))
#    w2 = rng_l2.standard_normal((w1.shape[1], l2_neurons))
#    b2 = rng_l2.standard_normal((1, l2_neurons))
#    w3 = rng_l3.standard_normal((w2.shape[1], l3_neurons))
#    b3 = rng_l3.standard_normal((1, l3_neurons))
#
# The ``default_rng`` is used to generate random numbers with reproducibility.

# %%
# Forward pass
# ------------
# The forward pass consists of set of calculations which are performed on the input until
# we get the output. In other words, it is the modification of input through the application
# of fully connected layers.
#
# .. code-block:: python
#
#    for e in range(epochs):
#
#        for batch_x, batch_y in batch_generator(X_train, y_train):
#
# Each epoch can consist of multiple batches. The actual number of batches in an
# epoch depends upon the number of examples and batch size. If we have 100 examples
# in our dataset and the batch size is 25, then we will have 4 batches. In this case
# the inner loop will have 4 iterations.

# %%
# hidden layer
# --------------
#
# .. math::
#
#     l1_{out} = sigmoid(inputs*w1 + b1)
#
# .. code-block:: python
#
#    def sigmoid(inputs):
#        return 1.0 / (1.0 + np.exp(-1.0 * inputs))
#
#    l1_out = np.dot(batch_x, w1) + b1
#    sig1_out = sigmoid(l1_out)
#

# %%
# second hidden layer
# --------------------
# The mathematics of second hidden layer is very much similar to first hidden layer.
# However, here we use the outputs from the first layer as inputs.
#
# .. math::
#
#      l2_{out} = sigmoid(l1_out * w2 + b2)
#
# .. code-block:: python
#
#    l2_out = np.dot(sig1_out, w2) + b2
#    sig2_out = sigmoid(l2_out)
#

# %%
# output layer
# -------------
# The output layer is also similar to other hidden layers, except that we are not applying
# any activation function here. Here we are performing a regression task. Had it been a
# classification problem, we would have been interested in using a relevant activation
# function here.
#
# .. math::
#
#      l3_{out} = l2_out * w3 + b3
#
# .. code-block:: python
#
#    l3_out = np.dot(sig2_out, w3) + b3
#

# %%
# loss calculation
# ------------------
# This step evaluate the performance of our model with current state of parameters. It provides us
# a scalar value whose value would like to reduce or minimize as a result of training process. The
# choice of loss function depends upon the task and objective. Here we are using mean squared error
# between true and predicted values as our loss function.
#
# .. code-block:: python
#
#    def mse(true, prediction):
#        return np.sum(np.power(prediction - true, 2)) / prediction.shape[0]
#
#    loss = mse(batch_y, out)
#
# The function ``mse`` calculate mean squared error between any two arrays. The first
# array is supposed to be the true values and second array will be the prediction obtained
# from the forward pass.

# %%
# Now we can write the how forward pass as below.
#
# .. code-block:: python
#
#    for e in range(epochs):
#
#        epoch_losses = []
#
#        for batch_x, batch_y in batch_generator(X_train, y_train):
#
#            # FORWARD PASS
#            l1_out = np.dot(batch_x, w1) + b1
#            sig1_out = sigmoid(l1_out)
#
#            l2_out = np.dot(sig1_out, w2) + b2
#            sig2_out = sigmoid(l2_out)
#
#            l3_out = np.dot(sig2_out, w3) + b3
#
#            # LOSS CALCULATION
#            loss = mse(batch_y, l3_out)
#            epoch_losses.append(loss)
#
# We are saving the loss obtained at each mini-batch step in a list ``epoch_losses``. This will
# be used later for plotting purpose.

# %%
# backward pass
# --------------
# We are interested in finding out how much loss is contributed by the each of the parameter
# in our neural network. So that we can tune/change the parameter accordingly. We have three
# layers and each layer has two kinds of parameters i.e.,
# weights and biases. Therefore, we would like to find out how much loss is contributed by weights
# and biases in these three layers. This can be achieved by finding out the partial derivative of
# the loss with respect to partial derivative of the specific parameter i.e., weights and biases.
#
# .. math::
#     h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x
#

# %%
# loss gradient
# ----------------
#
# .. code-block:: python
#
#    def mse_backward(true, prediction):
#        return 2.0 * (prediction - true) / prediction.shape[0]
#
#    d_loss = mse_backward(batch_y, l3_out)  # -> (batch_size, num_outs)
#

# %%
# third layer gradients
# ------------------------
# The third layer consisted of only two operations 1) dot product of inputs with ``w3`` and addition
# of `b3` bias in the result. Now during the backpropagation, we calculate three kinds of gradients
# 1) gradient of bias ``b3``, 2) gradient of weights ``w3`` and 3) gradient of inputs to 3rd layer
#
# .. code-block:: python
#
#    # bias third layer
#    d_b3 = np.sum(d_loss, axis=0, keepdims=True)  # -> (1, 1)
#
#    # weight third layer
#    input_grad_w3 = np.dot(d_loss, w3.T)  # -> (batch_size, l2_neurons)
#    d_w3 = np.dot(sig2_out.T, d_loss)  # -> (l2_neurons, num_outs)
#

# %%
# second layer gradients
# -------------------------
#
# .. code-block:: python
#
#    def sigmoid_backward(inputs, out_grad):
#        sig_out = sigmoid(inputs)
#        d_inputs = sig_out * (1.0 - sig_out) * out_grad
#        return d_inputs
#
#    # sigmoid second layer
#    # ((batch_size, batch_size), (batch_size, 14)) -> (batch_size, l2_neurons)
#    d_l2_out = sigmoid_backward(l2_out, input_grad_w3)
#
#    # bias second layer
#    d_b2 = np.sum(d_l2_out, axis=0, keepdims=True)  # -> (1, l2_neurons)
#
#    # weight second layer
#    input_grad_w2 = np.dot(d_l2_out, w2.T)  # -> (batch_size, l1_neurons)
#    d_w2 = np.dot(sig1_out.T, d_l2_out)  # -> (l1_neurons, l2_neurons)
#

# %%
# first layer gradients
# -------------------------
#
# .. code-block:: python
#
#    # ((batch_size, l1_neurons), (batch_size, l1_neurons)) -> (batch_size, l1_neurons)
#    d_sig1_out = sigmoid_backward(l1_out, input_grad_w2)
#
#    # bias first layer
#    d_b1 = np.sum(d_sig1_out, axis=0, keepdims=True)  # -> (1, l1_neurons)
#
#    # weight first layer
#    input_grads_w1 = np.dot(d_sig1_out, w1.T)  # -> (batch_size, num_ins)
#    # derivative of (the propagated loss) w.r.t weights
#    d_w1 = np.dot(batch_x.T, d_sig1_out)  # -> (num_ins, l1_neurons)
#

# %%
# parameter update
# -------------------------
# Now that we have calculated the gradients, we now know how much change needs to be
# carried out in each parameter (weights and biases). It is time to update weights and
# biases. This step is neural network libraries is carried out by the **optimizer``.
#
# .. code-block:: python
#
#    for e in range(epochs):
#
#        epoch_losses = []
#
#        for batch_x, batch_y in batch_generator(X_train, y_train):
#            # FORWARD PASS
#            ...
#
#            # BACKWARD PASS
#            ...
#
#            # OPTIMIZER STEP
#            w3 -= lr * d_w3
#            b3 -= lr * d_b3
#
#            w2 -= lr * d_w2
#            b2 -= lr * d_b2
#
#            w1 -= lr * d_w1
#            b1 -= lr * d_b1
#
# We can note that the parameter ``lr`` is kind of check on the change in parameters.
# Larger the value of ``lr``, the larger will be the change and vice versa.

# %%
# model evaluation
# -------------------------
#
# Once we have updated the parameters of the model i.e. we now have a new model, we
# would like to see how this new model performs. We do this by performing the forward
# pass with the updated parameters. However, this check is performed not on the training
# data but on a different data which is usually called validation data. We pass the inputs
# of the validation data through our network, calculate prediction and compare this prediction
# with true values to calculate a performance metric of our interest. Usually we would like
# to see the performance of our model on more than one performance metrics of different nature.
# The choice of performance metric is highly subjective to our task.
#
# .. code-block:: python
#
#     for e in range(epochs):
#
#         epoch_losses = []
#
#         for batch_x, batch_y in batch_generator(X_train, y_train):
#
#             ...
#         # Evaluation on validation data
#         l1_out_ = sigmoid(np.dot(x, w1) + b1)
#         l2_out_ = sigmoid(np.dot(l1_out_, w2) + b2)
#         predicted = np.dot(l2_out_, w3) + b3
#
#         val_loss = mse(y_val, predicted)
#

# %%
# loss curve
# -------------------------
# We will get the value of ``loss`` and ``val_loss`` after each mini-batch. We would
# be interesting in plotting these losses during the model training. We usually take
# the average of losses and val_losses during all the mini-batches in an epoch. Thus,
# after ``n`` epochs, we will have an array of length n  for loss and val_loss. Plotting
# these arrays together provides us important information about the training behaviour
# of our neural network.
#
# .. code-block:: python
#
#     from easy_mpl import plot
#
#     train_losses = np.full(epochs, fill_value=np.nan)
#     val_losses = np.full(epochs, fill_value=np.nan)
#
#     for e in range(epochs):
#
#         epoch_losses = []
#
#             for batch_x, batch_y in batch_generator(X_train, y_train):
#                 ...
#         # Evaluation on validation data
#         l1_out_ = sigmoid(np.dot(x, w1) + b1)
#         l2_out_ = sigmoid(np.dot(l1_out_, w2) + b2)
#         predicted = np.dot(l2_out_, w3) + b3
#
#         val_loss = mse(y_val, predicted)
#
#         train_losses[e] = np.nanmean(epoch_losses)
#         val_losses[e] = val_loss
#
#    plot(train_losses, label="Training", show=False)
#    plot(val_losses, label="Validation", grid=True)
#

# %%
# -------------------------
# To avoid repetition, we can put the forward pass of our network in a function so that
# can carry out this forward pass whenever we like, and calculate any desired performance
# metric for the prediction.
#
# .. code-block:: python
#
#    from SeqMetrics import RegressionMetrics
#
#    def eval_model(x, y, metric_name):
#        # Evaluation on validation data
#        l1_out_ = sigmoid(np.dot(x, w1) + b1)
#        l2_out_ = sigmoid(np.dot(l1_out_, w2) + b2)
#        prediction = np.dot(l2_out_, w3) + b3
#        metrics = RegressionMetrics(y, prediction)
#        return getattr(metrics, metric_name)()
#


# %%
#
# .. code-block:: python
#
#    from easy_mpl import plot
#
#    train_losses = np.full(epochs, fill_value=np.nan)
#    val_losses = np.full(epochs, fill_value=np.nan)
#
#    for e in range(epochs):
#
#        epoch_losses = []
#
#        for batch_x, batch_y in batch_generator(X_train, y_train):
#         ...
#        # Evaluation on validation data
#        val_loss = eval_model(batch_x, batch_y, 'mse')
#
#        train_losses[e] = np.nanmean(epoch_losses)
#        val_losses[e] = val_loss
#
#    plot(train_losses, label="Training", show=False)
#    plot(val_losses, label="Validation", grid=True)
#

# %%
# Now we can also evaluate model for any other performance metric
#
# .. code-block:: python
#
#    print(eval_model(batch_x, batch_y, 'nse'))
#

# %%
# early stopping
# -------------------------
#
# How long should we train our neural network i.e. what should be the value of epochs?
# The answer to this question can only be given by looking at the loss and validation loss curves.
# We should keep training our neural network as long as the performance of our network
# is improving on validation data i.e. as long as val_loss is decreasing. What if we
# have set the value of epochs to 5000 and the validation loss stops decreasing after 50th epoch?
# Should we wait for complete 5000 epochs to complete? We usually set a criteria to stop/break
# the training loop early depending upon the performance of our model on validation data. This is
# called early stopping. We following code, we break the training loop if the validation
# loss does not decrease for 50 consecutive epochs. The value of 50 is arbitrary here.
#
# .. code-block:: python
#
#
#    patience = 50
#    best_epoch = 0
#    epochs_since_best_epoch = 0
#
#    for e in range(epochs):
#
#        epoch_losses = []
#
#        for batch_x, batch_y in batch_generator(X_train, y_train):
#            ...
#
#        # calculation of val_loss
#            ...
#
#        if val_loss <= np.nanmin(val_losses):
#            epochs_since_best_epoch = 0
#            print(f"{e} {round(np.nanmean(epoch_losses).item(), 4)} {round(val_loss, 4)}")
#        else:
#            epochs_since_best_epoch += 1
#
#        if epochs_since_best_epoch > patience:
#            print(f"""Early Stopping at {e} because val loss did not improved since
#             {e - epochs_since_best_epoch}""")
#            break
#


# %%
# Complete code
# ---------------
# The complete python code which we have seen about is given as one peace below!

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from easy_mpl import plot, imshow
from numpy.random import default_rng
from SeqMetrics import RegressionMetrics
from ai4water.datasets import busan_beach
from ai4water.preprocessing import DataSet
from sklearn.preprocessing import StandardScaler

data = busan_beach()
#data = data.drop(data.index[317])
#data['tetx_coppml'] = np.log(data['tetx_coppml'])
print(data.shape)

s = StandardScaler()
data = s.fit_transform(data)

dataset = DataSet(data, val_fraction=0.0, seed=2809)
X_train, y_train = dataset.training_data()
X_val, y_val = dataset.test_data()

def batch_generator(X, Y, size=32):
    N = X.shape[0]

    for ii in range(0, N, size):
        X_batch, y_batch = X[ii:ii + size], Y[ii:ii + size]

        yield X_batch, y_batch


def sigmoid(inputs):
    return 1.0 / (1.0 + np.exp(-1.0 * inputs))

def sigmoid_backward(inputs, out_grad):
    sig_out = sigmoid(inputs)
    d_inputs = sig_out * (1.0 - sig_out) * out_grad
    return d_inputs

def relu(inputs):
    return np.maximum(0, inputs)

def relu_backward(inputs, out_grad):
    d_inputs = np.array(out_grad, copy = True)
    d_inputs[inputs <= 0] = 0
    return d_inputs

def mse(true, prediction):
    return np.sum(np.power(prediction - true, 2)) / prediction.shape[0]

def mse_backward(true, prediction):
    return 2.0 * (prediction - true) / prediction.shape[0]

def eval_model(x, y, metric_name)->float:
    # Evaluation on validation data
    l1_out_ = sigmoid(np.dot(x, w1) + b1)
    l2_out_ = sigmoid(np.dot(l1_out_, w2) + b2)
    prediction = np.dot(l2_out_, w3) + b3
    metrics = RegressionMetrics(y, prediction)
    return getattr(metrics, metric_name)()


# hyperparameters
lr = 0.001
epochs = 1000
l1_neurons = 10
l2_neurons = 5
l3_neurons = 1

# parameters (weights and biases)

scale = 1/max(1., (2+2)/2.)
limit = sqrt(3.0 * scale)
rng_l1 = default_rng(313)
rng_l2 = default_rng(313)
rng_l3 = default_rng(313)
w1 = rng_l1.standard_normal((dataset.num_ins, l1_neurons))
b1 = rng_l1.standard_normal((1, l1_neurons))
w2 = rng_l2.standard_normal((w1.shape[1], l2_neurons))
b2 = rng_l2.standard_normal((1, l2_neurons))
w3 = rng_l3.standard_normal((w2.shape[1], l3_neurons))
b3 = rng_l3.standard_normal((1, l3_neurons))

train_losses = np.full(epochs, fill_value=np.nan)
val_losses = np.full(epochs, fill_value=np.nan)

tolerance = 1e-5
patience = 50
best_epoch = 0
epochs_since_best_epoch = 0

for e in range(epochs):

    epoch_losses = []

    for batch_x, batch_y in batch_generator(X_train, y_train):

        # FORWARD PASS
        l1_out = np.dot(batch_x, w1) + b1
        sig1_out = sigmoid(l1_out)

        l2_out = np.dot(sig1_out, w2) + b2
        sig2_out = sigmoid(l2_out)

        l3_out = np.dot(sig2_out, w3) + b3

        # LOSS CALCULATION
        loss = mse(batch_y, l3_out)
        epoch_losses.append(loss)

        # BACKWARD PASS
        d_loss = mse_backward(batch_y, l3_out)  # -> (batch_size, num_outs)

        # bias third layer
        d_b3 = np.sum(d_loss, axis=0, keepdims=True)  # -> (1, 1)

        # weight third layer
        input_grad_w3 = np.dot(d_loss, w3.T)  # -> (batch_size, l2_neurons)
        d_w3 = np.dot(sig2_out.T, d_loss)  # -> (l2_neurons, num_outs)

        # sigmoid second layer
        # ((batch_size, batch_size), (batch_size, 14)) -> (batch_size, l2_neurons)
        d_l2_out = sigmoid_backward(l2_out, input_grad_w3)

        # bias second layer
        d_b2 = np.sum(d_l2_out, axis=0, keepdims=True)  # -> (1, l2_neurons)

        # weight second layer
        input_grad_w2 = np.dot(d_l2_out, w2.T)  # -> (batch_size, l1_neurons)
        d_w2 = np.dot(sig1_out.T, d_l2_out)  # -> (l1_neurons, l2_neurons)

        # sigmoid first layer
        # ((batch_size, l1_neurons), (batch_size, l1_neurons)) -> (batch_size, l1_neurons)
        d_sig1_out = sigmoid_backward(l1_out, input_grad_w2)

        # bias first layer
        d_b1 = np.sum(d_sig1_out, axis=0, keepdims=True)  # -> (1, l1_neurons)

        # weight first layer
        input_grads_w1 = np.dot(d_sig1_out, w1.T)  # -> (batch_size, num_ins)
        # derivate of (the propagated loss) w.r.t weights
        d_w1 = np.dot(batch_x.T, d_sig1_out)  # -> (num_ins, l1_neurons)

        # OPTIMIZER STEP
        w3 -= lr * d_w3
        b3 -= lr * d_b3

        w2 -= lr * d_w2
        b2 -= lr * d_b2

        w1 -= lr * d_w1
        b1 -= lr * d_b1

    # Evaluation on validation data
    val_loss = eval_model(X_val, y_val, "mse")

    train_losses[e] = np.nanmean(epoch_losses)
    val_losses[e] = val_loss

    if val_loss <= np.nanmin(val_losses):
        epochs_since_best_epoch = 0
        print(f"{e} {round(np.nanmean(epoch_losses).item(), 4)} {round(val_loss, 4)}")
    else:
        epochs_since_best_epoch += 1

    if epochs_since_best_epoch > patience:
        print(f"""Early Stopping at {e} because val loss did not improved since
         {e-epochs_since_best_epoch}""")
        break


plot(train_losses, label="Training", show=False)
plot(val_losses, label="Validation", grid=True)

print(eval_model(X_val, y_val, "r2"))

_, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 6), gridspec_kw={"hspace": 0.4})
imshow(w1, aspect="auto", colorbar=True, title="Layer 1 Weights", ax=ax1, show=False)
imshow(w2, aspect="auto", colorbar=True, title="Layer 2 Weights", ax=ax2, show=False)
plot(w3, ax=ax3, show=False, title="Layer 3 Weights")
plt.show()

_, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 6), gridspec_kw={"hspace": 0.4})
imshow(d_w1, aspect="auto", colorbar=True, title="Layer 1 Gradients", ax=ax1, show=False)
imshow(d_w2, aspect="auto", colorbar=True, title="Layer 2 Gradients", ax=ax2, show=False)
plot(d_w3, ax=ax3, show=False, title="Layer 3 Gradients")
plt.show()


"""
===================
ANN in numpy
===================
"""

# %%imports
#
# .. code-block:: python
#
#    import numpy as np
#

# %%
# data preparation
# ------------------
#
# .. code-block:: python
#
#     from ai4water.datasets import busan_beach
#     data = busan_beach()
#

# %%
# splitting
# -----------
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

# %%
# hyperparameters
#
# .. code-block:: python
#
#    lr = 0.01
#    epochs = 1000
#    l1_neurons = 10
#    l2_neurons = 5
#    l3_neurons = 1
#

# %%
# weights and biases
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

# %%
# Forwad pass
# ------------
#
# .. code-block:: python
#
#    for e in range(epochs):
#
#        for batch_x, batch_y in batch_generator(X_train, y_train):
#

# %%
## hidden layer
# --------------
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
#
# .. code-block:: python
#
#    l2_out = np.dot(sig1_out, w2) + b2
#    sig2_out = sigmoid(l2_out)
#

# %%
# output
# --------
#
# .. code-block:: python
#
#    l3_out = np.dot(sig2_out, w3) + b3
#

# %%
# loss calculation
# ------------------
#
# .. code-block:: python
#
#    def mse(true, prediction):
#        return np.sum(np.power(prediction - true, 2)) / prediction.shape[0]
#
#    loss = mse(batch_y, out)
#

# %%
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

# %%
# backward pass
# --------------
# We are interested in finding out how much loss is contributed by the each of the parameter
# in our neural network. We have three layers and each layer has two kinds of parameters i.e.,
# weights and biases. Therefore, we would like to find out how much loss is contributed by weights
# and biases in these three layers. This can be acheived by finding out the partial derivate of
# the loss with respect to partial derivate of the specific parameter i.e., weights and biases.
#
# h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x
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
# The third layer consisted of only two operations 1) dot product of inputs with `w3` and addition
# of `b3` bias in the result. Now during the backpropagation, we calculate three kinds of gradients
# 1) gradient of bias `b3`, 2) gradient of weights `w3` and 3) gradient of inputs to 3rd layer
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
#    # derivate of (the propagated loss) w.r.t weights
#    d_w1 = np.dot(batch_x.T, d_sig1_out)  # -> (num_ins, l1_neurons)
#

# %%
# parameter update
# -------------------------
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

# %%
# model evaluation
# -------------------------
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
# We can also evaluate model for any other peformance metric
#
# .. code-block:: python
#
#    print(eval_model(batch_x, batch_y, 'nse'))
#

# %%
# early stopping
# -------------------------
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




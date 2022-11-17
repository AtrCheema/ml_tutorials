"""
==================================
understanding Dense layer in Keras
==================================

This notebook describes dense layer or fully connected layer using tensorflow.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

##############################################

def reset_seed(seed=313):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

np.set_printoptions(linewidth=100, suppress=True)

print(tf.__version__)

##############################################
print(np.__version__)

##############################################
# set some global parameters

input_features = 2
batch_size = 10
dense_units = 5

##############################################
# define input to model

in_np = np.random.randint(0, 100, size=(batch_size,input_features))
print(in_np)

##############################################
# build a model consisting of single dense layer

##############################################

reset_seed()


ins = Input(input_features, name='my_input')
out = Dense(dense_units, use_bias=False, name='my_output')(ins)
model = Model(inputs=ins, outputs=out)

##############################################

out_np = model.predict(in_np)

print(out_np)

##############################################

print(out_np.shape)

##############################################
# We can get all layers of model as list

##############################################

print(model.layers)

##############################################
# or a specific layer by its name

##############################################

dense_layer = model.get_layer('my_output')

##############################################
# input to dense layer must be of the shape

print(dense_layer.input_shape)

##############################################
# output from dense layer will be of the shape

print(dense_layer.output_shape)

##############################################
# dense layer ususally has two variables i.e. weight/kernel and bias. As we did
# not use bias thus no bias is shown

print(dense_layer.weights)

##############################################
# The shape of the dense weights is of the form `(input_size, units)`
# `dense_layer.weights` returns a list, the first variable of which kernel/weights.
# We can convert a numpy version of weights

##############################################

dense_w = dense_layer.weights[0].numpy()
print(dense_w.shape)

##############################################

print(dense_w)

##############################################
# The output from our model consisting of a single dense layer is simply the matrix
# multiplication between input and weight matrix as can be verified from below.

np.matmul(in_np, dense_w)

##############################################
# compare above output from the model's output which was obtained earlier.

##############################################
# Using Bias
#============
# By default the `Dense` layer in tensorflow uses bias as well.

##############################################

reset_seed()
tf.keras.backend.clear_session()

ins = Input(input_features, name='my_input')
out = Dense(5, use_bias=True,  name='my_output')(ins)
model = Model(inputs=ins, outputs=out)

##############################################

out_np = model.predict(in_np)
print(out_np.shape)
print(out_np)

##############################################

dense_layer = model.get_layer('my_output')
print(dense_layer.weights)

##############################################
# The bias vector above was all zeros thus had no effect on model output as the
# equation for dense layer becomes
# $$ y = Ax + b$$
# We can initialize bias vector with ones and see the output

##############################################

reset_seed()

ins = Input(input_features, name='my_input')
out = Dense(dense_units, use_bias=True, bias_initializer='ones', name='my_output')(ins)
model = Model(inputs=ins, outputs=out)

##############################################

out_np = model.predict(in_np)
print(out_np.shape)
print(out_np)

##############################################

dense_layer = model.get_layer('my_output')
print(dense_layer.weights)

##############################################
# We can verify that the model's output is obtained following the equation we
# wrote above.

##############################################

dense_layer = model.get_layer('my_output')
dense_w = dense_layer.weights[0].numpy()
np.matmul(in_np, dense_w) + np.ones(dense_units)

##############################################
# using `activation` function
#=============================
# We can add non-linearity to the output of dense layer by making use of `activation`
# keyword argument. A common `activation` function is `relu` which makes all
# the values below 0 as zero.
# In this case the equation of dense layer will become
# $$ y = \alpha (Ax + b) $$
# Where $\alpha$ is the non-linearity applied.

##############################################

reset_seed()

ins = Input(input_features, name='my_input')
out = Dense(dense_units, use_bias=True, bias_initializer='ones',
            activation='relu', name='my_output')(ins)
model = Model(inputs=ins, outputs=out)

out_np = model.predict(in_np)
print(out_np.shape)
print(out_np)

##############################################
# We can again verify that the above output from dense layer follows the equation
# that we wrote above.

##############################################

def relu(X):
   return np.maximum(0,X)


dense_layer = model.get_layer('my_output')
dense_w = dense_layer.weights[0].numpy()
relu(np.matmul(in_np, dense_w) + np.ones(dense_units))

##############################################
# customizing weights
#=========================
# we can set the weights and bias of dense layer to values of our choice. This is
# useful for example when we want to initialize the weights/bias with the values
# that we already have.

##############################################

custom_dense_weights = np.array([[1, 2, 3 , 4,  5],
                                 [6, 7, 8 , 9 , 10]], dtype=np.float32)
custom_bias = np.array([0., 0., 0., 0., 0.])

reset_seed()

ins = Input(input_features, name='my_input')

dense_lyr = Dense(dense_units, use_bias=True, bias_initializer='ones', name='my_output')
out = dense_lyr(ins)

model = Model(inputs=ins, outputs=out)

dense_lyr.set_weights([custom_dense_weights, custom_bias])

##############################################
# The method `set_weights` must be called after initializing `Model` class.
# The input to `set_weights` is a list containing both weight matrix and bias
# vector respectively.

##############################################

out_np = model.predict(in_np)
print(out_np.shape)
print(out_np)

##############################################

dense_layer = model.get_layer('my_output')
dense_w = dense_layer.weights[0].numpy()
print(dense_w)

##############################################
# Verify that the output from dense is just matrix multiplication.

np.matmul(in_np, custom_dense_weights) + np.zeros(dense_units)

##############################################
# Reducing Dimensions
#=========================
# Dense layer can be used to reduce last dimension of incoming input.
# In following the size is reduced from `(10, 20, 30)` ==> `(10, 20, 1)`

##############################################

input_shape = 20, 30
in_np = np.random.randint(0, 100, size=(batch_size,*input_shape))

reset_seed()


ins = Input(input_shape, name='my_input')
out = Dense(1, use_bias=False, name='my_output')(ins)
model = Model(inputs=ins, outputs=out)
out_np = model.predict(in_np)
print('input shape: {}\n output shape: {}'.format(in_np.shape, out_np.shape))
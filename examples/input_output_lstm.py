"""
==================================
understanding Input/output of LSTM
==================================

The purpose of this notebook to determine the input and output shapes of LSTM
in keras/tensorflow. It also shows how the output changes when we use different
options such as ``return_sequences`` and ``return_state``
arguments in LSTM/RNN layers of tensorflow/keras.
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling1D, Flatten, Conv1D
from tensorflow.keras.layers import Input, LSTM, Reshape, TimeDistributed

# to suppress scientific notation while printing arrays
np.set_printoptions(suppress=True)

def reset_graph(seed=313):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

tf.__version__

##############################################

seq_len = 9
in_features = 3
batch_size = 2
units = 5

# define input data
data = np.random.normal(0,1, size=(batch_size, seq_len, in_features))
print('input shape is', data.shape)


##############################################

reset_graph()

##############################################
# Input to LSTM
#------------------------------

# The input to LSTM is 3D where each dimension is expected to have following meaning
# (batch_size, sequence_length, num_inputs)
# the batch_size determines the number of samples, sequence_legth determines the length
# of historical/temporal data used by LSTM and num_inputs is the number of input features

# define model
inputs1 = Input(shape=(seq_len, in_features))
lstm1 = LSTM(units)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)
model.inputs

##############################################
# Output from LSTM
#------------------------------

# In Keras, the output from LSTM is 2D and each dimension has following meaning
# (batch_size, units)
# the units here represents the number of units/neuron of LSTM layer.

# check output
output = model.predict(data)
print('output shape is ', output.shape)
print(output)

##############################################
# Return Sequence
#------------------------------
# If we use ``return_sequences=True``, we can get hidden state which is also output,
# at each time step instead of just one final output.

##############################################

reset_graph()

print('input shape is', data.shape)

# define model
inputs1 = Input(shape=(seq_len, in_features))
lstm1 = LSTM(units, return_sequences=True)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)

# check output
output = model.predict(data)
print('output shape is ', output.shape)
print(output)

##############################################
# Return States
#--------------------
# If we use ``return_state=True``, it will give final hidden state/output plus the cell state as well

##############################################

reset_graph()

# define model
inputs1 = Input(shape=(seq_len, in_features))
lstm1, state_h, state_c = LSTM(units, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

# check output
_h, h, c = model.predict(data)
print('_h: shape {} values \n {}\n'.format(_h.shape, _h))
print('h: shape {} values \n {}\n'.format(h.shape, h))
print('c: shape {} values \n {}'.format(c.shape, c))

##############################################
# using both at same time
# We can use both ``return_sequences`` and ``return_states`` at same time as well.

##############################################

reset_graph()

# define model
inputs1 = Input(shape=(seq_len, in_features))
lstm1, state_h, state_c = LSTM(units, return_state=True, return_sequences=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

# check output
_h, h, c = model.predict(data)
print('_h: shape {} values \n {}\n'.format(_h.shape, _h))
print('h: shape {} values \n {}\n'.format(h.shape, h))
print('c: shape {} values \n {}'.format(c.shape, c))

##############################################
# time major
#--------------------
# By ``time_major`` we mean that the last dimention i.e. 3rd dimension represents time
# and the second last represents input features. Thus the 3D input to lstm will become
# (batch_size, num_inputs, sequence_length)

reset_graph()

# define model
inputs1 = Input(shape=(in_features, seq_len))
lstm1 = LSTM(units, time_major=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1])
model.inputs

##############################################

# we will have to shift the dimensions of numpy array to make it time_major
# check output
time_major_data = np.moveaxis(data, [1,2], [2,1])
time_major_data.shape

##############################################

h = model.predict(time_major_data)
print('h: shape {} values \n {}\n'.format(h.shape, h))

##############################################
# CNN -> LSTM
#------------------------
# We can append LSTM with any other layer. The only requirement is that the output
# from that layer should match the input requirement of LSTM i.e. the output from the
# layer that we want to add before LSTM should be 3D of shape (batch_size, num_inputs, seq_length)

reset_graph()

# define model
inputs = Input(shape=(seq_len, in_features))
cnn = Conv1D(filters=2, kernel_size=2, padding="same")(inputs)
max_pool = MaxPooling1D(padding="same")(cnn)
max_pool

##############################################
# as the shape of ``max_pool`` tensor matches the input requirement of LSTM we
# can combine it with LSTM

h = LSTM(units)(max_pool)
model = Model(inputs=inputs, outputs=h)
model.summary()

##############################################
# However, this is not how CNN is comined with LSTM at its start. The purpose is
# usually to break the sequence length into small sub-sequences and then apply the
# **same** CNN on those sub-sequences. We can achieve this as following

sub_sequences = 3

reset_graph()
# define model
inputs = Input(shape=(seq_len, in_features))
time_steps = seq_len // sub_sequences
reshape = Reshape(target_shape=(sub_sequences, time_steps, in_features))(inputs)
cnn = TimeDistributed(Conv1D(filters=2, kernel_size=2, padding="same"))(reshape)
max_pool = TimeDistributed(MaxPooling1D(padding="same"))(cnn)
flatten = TimeDistributed(Flatten())(max_pool)
flatten

##############################################
# the shape of ``flatten`` tensor again matches the input requirements of LSTM so
# we can again attach LSTM after it.

h = LSTM(units)(flatten)
model = Model(inputs=inputs, outputs=h)
model.summary()

##############################################
# LSTM -> 1D CNN
#------------------------
# We can put 1d cnn at the end of LSTM to further extract some features from LSTM output.

##############################################

reset_graph()

print('input shape is', data.shape)

# define model
inputs = Input(shape=(seq_len, in_features))
lstm_layer = LSTM(units, return_sequences=True)
lstm_outputs = lstm_layer(inputs)
print('lstm output: ', lstm_outputs.shape)

conv1 = Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_len, units))(lstm_outputs)
print('conv output: ', conv1.shape)

max1d1 = MaxPooling1D(pool_size=2)(conv1)
print('max pool output: ', max1d1.shape)

flat1 = Flatten()(max1d1)
print('flatten output: ', flat1.shape)

model = Model(inputs=inputs, outputs=flat1)

# check output
output = model.predict(data)
print('output shape: ', output.shape)

##############################################
# The output from LSTM/RNN looks roughly as below.
# $$ h_t = tanh(b + Wh_{t-1} + UX_t) $$


##############################################
# weights of our input against every neuron in LSTM

print('kernel U: ', lstm_layer.get_weights()[0].shape)

##############################################
# weights of our hidden state a.k.a the output of LSTM in the
# previous timestep (t-1) against every neuron in LSTM

print('recurrent kernel, W: ', lstm_layer.get_weights()[1].shape)

##############################################
print('bias: ', lstm_layer.get_weights()[2].shape)

##############################################
# This post is inspired from Jason Brownlee's [page](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)
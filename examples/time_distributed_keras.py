"""
==================================
understanding Dense layer in Keras
==================================

"""

# simple `Conv1D`

##############################################

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Conv1D, LSTM, MaxPool1D
import numpy as np

def reset_seed(seed=313):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

np.set_printoptions(linewidth=150)

print(tf.__version__, np.__version__)

##############################################

input_features = 3
lookback = 6
batch_size=2
input_shape = lookback,input_features
ins = Input(shape=input_shape, name='my_input')
outs = Conv1D(filters=8, kernel_size=3,
              strides=1, padding='same', kernel_initializer='ones',
              name='my_conv1d')(ins)
model = Model(inputs=ins, outputs=outs)

input_array = np.arange(36).reshape((batch_size, *input_shape))
conv1d_weights = model.get_layer('my_conv1d').weights[0].numpy()
output_array = model.predict(input_array)

##############################################

print(input_array.shape)

print(input_array)

##############################################

print(conv1d_weights.shape)

print(conv1d_weights)

##############################################

print(output_array)

print(output_array.shape)

##############################################

# multiple inputs multiple layers

##############################################


input_features = 3
lookback = 3
batch_size=2
input_shape = lookback,input_features
ins1 = Input(shape=input_shape, name='my_input1')
ins2 = Input(shape=input_shape, name='my_input2')
outs1 = Conv1D(filters=8, kernel_size=3,
              strides=1, padding='same', kernel_initializer='ones',
              name='my_conv1d1')(ins1)
outs2 = Conv1D(filters=8, kernel_size=3,
              strides=1, padding='same', kernel_initializer='ones',
              name='my_conv1d2')(ins2)
model = Model(inputs=[ins1, ins2], outputs=[outs1, outs2])

sub_seq = 2
input_shape = sub_seq, 3, input_features
input_array = np.arange(36).reshape((batch_size, *input_shape))
input_array1 = input_array[:, 0, :, :]
input_array2 = input_array[:, 1, :, :]

conv1d1_weights = model.get_layer('my_conv1d1').weights[0].numpy()
conv1d2_weights = model.get_layer('my_conv1d2').weights[0].numpy()
output_array = model.predict([input_array1, input_array2])

##############################################

print(input_array1, '\n\n', input_array2)

##############################################
print(conv1d2_weights)

print(conv1d2_weights)

##############################################

print(output_array)

##############################################

# multiple inputs shared layer

##############################################


input_features = 3
lookback = 6
sub_seq = 2
input_shape = sub_seq, 3, input_features
batch_size = 2

ins = Input(shape=input_shape, name='my_input')
conv = Conv1D(filters=8, kernel_size=3,
              strides=1, padding='same',
              kernel_initializer='ones', name='my_conv1d')

conv1_out = conv(ins[:, 0, :, :])
conv2_out = conv(ins[:, 1, :, :])
model = Model(inputs=ins, outputs=[conv1_out, conv2_out])

input_array = np.arange(36).reshape((batch_size, *input_shape))
conv1d_weights = model.get_layer('my_conv1d').weights[0].numpy()
output_array = model.predict(input_array)

##############################################

print(input_array)

##############################################

print(conv1d_weights)

##############################################

print(output_array[0])

##############################################

print(output_array[1])

##############################################

# `TimeDistributed Conv1D`

##############################################


input_features = 3
lookback = 6
sub_seq = 2
input_shape = sub_seq, 3, input_features
batch_size = 2

ins = Input(shape=input_shape, name='my_input')
outs = TimeDistributed(Conv1D(filters=8, kernel_size=3,
              strides=1, padding='same', kernel_initializer='ones',
              name='my_conv1d'))(ins)
model = Model(inputs=ins, outputs=outs)

input_array = np.arange(36).reshape((batch_size, *input_shape))
output_array = model.predict(input_array)

##############################################

print(input_array)

##############################################

print(output_array)

##############################################

# So `TimeDistributed` Just applies same `Conv1D` to each sub-sequence/incoming input.

# `TimeDistributed` `LSTM`

##############################################


tf.random.set_seed(313)

input_features = 3
sub_seq = 2
input_shape = sub_seq, 3, input_features
batch_size = 2

ins = Input(shape=input_shape, name='my_input')
outs = TimeDistributed(LSTM(units=8, name='my_lstm'))(ins)
model = Model(inputs=ins, outputs=outs)

input_array = np.arange(36).reshape((batch_size, *input_shape))
output_array = model.predict(input_array)

##############################################

print(input_array)

##############################################

print(output_array[:, 0, :])

##############################################

print(output_array[:, 1, :])

##############################################

# manual weight sharing of `LSTM`

##############################################


tf.random.set_seed(313)

input_features = 3
sub_seq = 2
input_shape = sub_seq, 3, input_features
batch_size = 2

ins = Input(shape=input_shape, name='my_input')
lstm = LSTM(units=8, name='my_lstm')

lstm1_out = lstm(ins[:, 0, :, :])
lstm2_out = lstm(ins[:, 1, :, :])
model = Model(inputs=ins, outputs=[lstm1_out, lstm2_out])

input_array = np.arange(36).reshape((batch_size, *input_shape))
output_array = model.predict(input_array)

##############################################

print(input_array)

##############################################

print(output_array[0])

##############################################

print(output_array[1])

##############################################

# Curious case of `Dense`

##############################################


tf.random.set_seed(313)

input_features = 3
lookback = 6
batch_size=2
input_shape = lookback,input_features

input_shape = lookback, input_features
ins = Input(input_shape, name='my_input')
out = Dense(units=5,  name='my_output')(ins)
model = Model(inputs=ins, outputs=out)

input_array = np.arange(36).reshape(batch_size, *input_shape)
output_array = model.predict(input_array)

##############################################

print(input_array.shape)

##############################################

print(input_array)

##############################################
print(output_array.shape)

##############################################

print(output_array)

##############################################


tf.random.set_seed(313)

input_features = 3
lookback = 6
sub_seq = 2
input_shape = sub_seq, 3, input_features
batch_size = 2

ins = Input(input_shape, name='my_input')
out = TimeDistributed(Dense(units=5, name='my_output'))(ins)
model = Model(inputs=ins, outputs=out)


input_array = np.arange(36).reshape(batch_size, *input_shape)
output_array = model.predict(input_array)

##############################################

print(input_array.shape)

##############################################

print(input_array)

##############################################

print(output_array.shape)

##############################################

print(output_array)

##############################################

# so far looks very similar to `TimeDistributed(Conv1D)` or `TimeDistributed(LSTM)`.


##############################################


tf.random.set_seed(313)

input_features = 3
lookback = 6
input_shape = lookback, input_features
batch_size = 2

ins = Input(input_shape, name='my_input')
out = TimeDistributed(Dense(5, use_bias=False, name='my_output'))(ins)
model = Model(inputs=ins, outputs=out)

input_array = np.arange(36).reshape(batch_size, *input_shape)
output_array = model.predict(input_array)

##############################################

print(input_array.shape)

print(input_array)

##############################################

print(output_array.shape)

print(output_array)

##############################################

# So whether we we `TimeDistributed(Dense)` or `Dense`, they are actually equivalent.

# What if we try same with `Conv1D` or `LSTM` i.e. wrapping these layers in
# `TimeDistributed` without modifying/dividing input into sub-sequences?

##############################################


input_features = 3
lookback = 6
input_shape = lookback, input_features

# uncomment following lines

# ins = Input(shape=(lookback, input_features), name='my_input')
# outs = TimeDistributed(Conv1D(filters=8, kernel_size=3,
#               strides=1, padding='valid', kernel_initializer='ones',
#               name='my_conv1d'))(ins)
# model = Model(inputs=ins, outputs=outs)

input_array = np.arange(36).reshape((batch_size, *input_shape))


##############################################
# The above error message can be slightly confusing or atleast can be resolved
# in a wrong manner as we do in following case;

##############################################


input_features = 3
lookback = 6
input_shape = lookback, input_features
ins = Input(shape=(batch_size, lookback, input_features), name='my_input')
outs = TimeDistributed(Conv1D(filters=8, kernel_size=3,
              strides=1, padding='valid', kernel_initializer='ones',
              name='my_conv1d'))(ins)
model = Model(inputs=ins, outputs=outs)

input_array = np.arange(36).reshape((batch_size, *input_shape))
print(input_array.shape)

##############################################

# So we are able to compile the model, although it is wrong.

##############################################

# uncomment following 2 lines
# output_array = model.predict(input_array)
# print(output_array.shape)

##############################################

# This error message is exactly related to `TimeDistributed` layer. The `TimeDistributed`
# layer here expects input having 4 dimensions, 1st being batch size, second being the
# sub-sequences, 3rd being the time-steps or whatever and 4rth being number of input
# features here.

# Anyhow, the conclusion is, we can't just wrap layers in `TimeDistributed` except
# for `Dense` layer. Hence, using `TimeDistributed(Dense)` does not make any
# sense (to me until version 2.3.0).

##############################################



# More than just weight sharing

# `TimeDistributed` layer is meant to provide more functionality than just weight
# sharing. We see, pooling layers or flatten layers wrapped into `TimeDistributed`
# layer even though pooling layers or flattening layers don't have any weights.
# This is because if we have applied `TimeDistributed(Conv1D)`, this will sprout
# output for each sub-sequence. We would naturally like to apply pooling and
# consequently flattening layers to each output fo the sub-sequences.

##############################################


input_features = 3
sub_seq = 2
input_shape = sub_seq, 3, input_features
batch_size = 2

ins = Input(shape=input_shape, name='my_input')
conv_outs = TimeDistributed(Conv1D(filters=8, kernel_size=3,
              strides=1,
              padding='same',
              kernel_initializer='ones',
              name='my_conv1d'))(ins)
outs = TimeDistributed(MaxPool1D(pool_size=2))(conv_outs)
model = Model(inputs=ins, outputs=[outs, conv_outs])

input_array = np.arange(36).reshape((batch_size, *input_shape))
output_array, conv_output = model.predict(input_array)

##############################################

print(input_array.shape)

##############################################

print(input_array)

##############################################

print(conv_output.shape)

##############################################

print(conv_output)

##############################################

print(output_array.shape)

##############################################

print(output_array)

##############################################


input_features = 3
sub_seq = 2
input_shape = sub_seq, 3, input_features
batch_size = 2

ins = Input(shape=input_shape, name='my_input')
conv_outs = TimeDistributed(Conv1D(filters=8, kernel_size=3,
              strides=1, padding='same',
              kernel_initializer='ones',
              name='my_conv1d'))(ins)
outs = TimeDistributed(MaxPool1D(pool_size=2, padding='same'))(conv_outs)
model = Model(inputs=ins, outputs=outs)

input_array = np.arange(36).reshape((batch_size, *input_shape))
output_array = model.predict(input_array)

##############################################

print(input_array.shape)

##############################################

print(input_array)


##############################################

print(output_array.shape)

##############################################

print(output_array)

##############################################
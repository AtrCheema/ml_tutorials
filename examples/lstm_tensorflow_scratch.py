"""
============================================
Implementing LSTM in tensorflow from scratch
============================================

The purpose of this notebook is to illustrate how to build an LSTM from scratch in Tensorflow.
Although the Tensorflow has implementation of LSTM in Keras. But since it comes with a lot of
implementation options, reading the code of Tensorflow for LSTM can be confusing at the start.
Therefore here is vanilla implementation of LSTM in Tensorflow. It has been shown that the
results of this vanilla LSTM are full reproducible with Keras'LSTM. This shows that the
simple implementation of LSTM in Tensorflow just has four equations and a for loop through time.

"""

import os
import random

import numpy as np
np.__version__


##############################################

import tensorflow as tf
tf.__version__

##############################################

def seed_all(seed):
    """reset seed for reproducibility"""

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if int(tf.__version__.split('.')[0]) == 1:
        tf.compat.v1.random.set_random_seed(seed)
    elif int(tf.__version__.split('.')[0]) > 1:
        tf.random.set_seed(seed)


from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.layers import LSTM as KLSTM
from tensorflow.keras.models import Model
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K

##############################################

assert tf.__version__ > "2.1", "results are not reproducible with Tensorflow below 2"

##############################################

num_inputs = 3  # number of input features
lstm_units = 32
lookback_steps = 5  # also known as time_steps or sequence length
num_samples = 10  # length of x,y


class SimpleLSTM(Layer):
    """A simplified implementation of LSTM layer with keras
    """

    def __init__(self, units, **kwargs):

        super(SimpleLSTM, self).__init__(**kwargs)

        self.activation = tf.nn.tanh
        self.rec_activation = tf.nn.sigmoid
        self.units = units

    def call(self, inputs):

        initial_state = tf.zeros((10, self.units))  # todo

        last_output, outputs, states = K.rnn(
            self.cell,
            inputs,
            [initial_state, initial_state]
        )

        return last_output

    def cell(self, inputs, states):

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)

        x_i = K.dot(inputs, k_i)
        x_f = K.dot(inputs, k_f)
        x_c = K.dot(inputs, k_c)
        x_o = K.dot(inputs, k_o)

        i = self.rec_activation(x_i + K.dot(h_tm1, self.rec_kernel[:, :self.units]))

        f = self.rec_activation(x_f + K.dot(h_tm1, self.rec_kernel[:, self.units:self.units * 2]))

        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.rec_kernel[:, self.units * 2:self.units * 3]))

        o = self.rec_activation(x_o + K.dot(h_tm1, self.rec_kernel[:, self.units * 3:]))

        h = o * self.activation(c)

        return h, [h, c]

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer="glorot_uniform")

        self.rec_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer="orthogonal")

        self.built = True

        return


##############################################

inputs_tf = tf.range(150, dtype=tf.float32)
inputs_tf = tf.reshape(inputs_tf, (num_samples, lookback_steps, num_inputs))

seed_all(313)
lstm = SimpleLSTM(lstm_units)
h1 = lstm(inputs_tf)
h1_sum = tf.reduce_sum(h1)
print(K.eval(h1_sum))


##############################################
# Now check the results of original lstm of Keras

seed_all(313)
lstm = KLSTM(lstm_units,
             recurrent_activation="sigmoid",
             unit_forget_bias=False,
             use_bias=False,
            )
h2 = lstm(inputs_tf)
h2_sum = tf.reduce_sum(h2)
print(K.eval(h2_sum))

###########################################
# with bias
#------------------------------------------


class LSTMWithBias(Layer):
    """A simplified implementation of LSTM layer with keras
    """

    def __init__(self, units, use_bias=True, **kwargs):

        super(LSTMWithBias, self).__init__(**kwargs)

        self.activation = tf.nn.tanh
        self.rec_activation = tf.nn.sigmoid
        self.units = units
        self.use_bias = use_bias

    def call(self, inputs):

        initial_state = tf.zeros((10, self.units))  # todo

        last_output, outputs, states = K.rnn(
            self.cell,
            inputs,
            [initial_state, initial_state]
        )

        return last_output

    def cell(self, inputs, states):

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)

        x_i = K.dot(inputs, k_i)
        x_f = K.dot(inputs, k_f)
        x_c = K.dot(inputs, k_c)
        x_o = K.dot(inputs, k_o)

        if self.use_bias:
            b_i, b_f, b_c, b_o = array_ops.split(
                self.bias, num_or_size_splits=4, axis=0)
            x_i = K.bias_add(x_i, b_i)
            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        i = self.rec_activation(x_i + K.dot(h_tm1, self.rec_kernel[:, :self.units]))

        f = self.rec_activation(x_f + K.dot(h_tm1, self.rec_kernel[:, self.units:self.units * 2]))

        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.rec_kernel[:, self.units * 2:self.units * 3]))

        o = self.rec_activation(x_o + K.dot(h_tm1, self.rec_kernel[:, self.units * 3:]))

        h = o * self.activation(c)

        return h, [h, c]

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer="zeros")

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer="glorot_uniform")

        self.rec_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer="orthogonal")

        self.built = True
        return


##############################################

seed_all(313)

seed_all(313)
lstm = LSTMWithBias(lstm_units)
h1 = lstm(inputs_tf)
h1_sum = tf.reduce_sum(h1)
print(K.eval(h1_sum))


##############################################

seed_all(313)
lstm = KLSTM(lstm_units,
             recurrent_activation="sigmoid",
             unit_forget_bias=False)
h2 = lstm(inputs_tf)
h2_sum = tf.reduce_sum(h2)
print(K.eval(h2_sum))


##############################################
# implementing temporal loop
#---------------------------------------------

# so far we had been using k.rnn() function to implement the temporal (for) loop
# of LSTM. Let's see what is inside it!

class LSTM(Layer):
    """A simplified implementation of LSTM layer with keras
    """

    def __init__(self, units, use_bias=True, **kwargs):

        super(LSTM, self).__init__(**kwargs)

        self.activation = tf.nn.tanh
        self.rec_activation = tf.nn.sigmoid
        self.units = units
        self.use_bias = use_bias

    def call(self, inputs, **kwargs):

        initial_state = tf.zeros((10, self.units))  # todo

        inputs = tf.transpose(inputs, [1, 0, 2])
        lookback, _, _ = inputs.shape
        state = [initial_state, initial_state]

        outputs, states = [], []
        for time_step in range(lookback):

            _out, state = self.cell(inputs[time_step], state)

            outputs.append(_out)
            states.append(state)

        outputs = tf.stack(outputs)
        states = tf.stack(states)

        outputs = tf.transpose(outputs, [1, 0, 2])

        last_output = outputs[:, -1]

        return last_output

    def cell(self, inputs, states):

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)

        x_i = K.dot(inputs, k_i)
        x_f = K.dot(inputs, k_f)
        x_c = K.dot(inputs, k_c)
        x_o = K.dot(inputs, k_o)

        if self.use_bias:
            b_i, b_f, b_c, b_o = array_ops.split(
                self.bias, num_or_size_splits=4, axis=0)
            x_i = K.bias_add(x_i, b_i)
            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        i = self.rec_activation(x_i + K.dot(h_tm1, self.rec_kernel[:, :self.units]))

        f = self.rec_activation(x_f + K.dot(h_tm1, self.rec_kernel[:, self.units:self.units * 2]))

        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.rec_kernel[:, self.units * 2:self.units * 3]))

        o = self.rec_activation(x_o + K.dot(h_tm1, self.rec_kernel[:, self.units * 3:]))

        h = o * self.activation(c)

        return h, [h, c]

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer="zeros")

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer="glorot_uniform")

        self.rec_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer="orthogonal")

        self.built = True
        return

##############################################

seed_all(313)
lstm = LSTM(lstm_units)
h1 = lstm(inputs_tf)
h1_sum = tf.reduce_sum(h1)
print(K.eval(h1_sum))


##############################################

seed_all(313)
lstm = KLSTM(lstm_units,
             recurrent_activation="sigmoid",
             unit_forget_bias=False)
h2 = lstm(inputs_tf)
h2_sum = tf.reduce_sum(h2)
print(K.eval(h2_sum))


##############################################
# adding some more options
#--------------------------------------------


class LSTM(Layer):
    """A simplified implementation of LSTM layer with keras
    """

    def __init__(
            self,
            units,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            return_state=False,
            return_sequences=False,
            time_major=False,
            ** kwargs
    ):

        super(LSTM, self).__init__(**kwargs)

        self.activation = tf.nn.tanh
        self.rec_activation = tf.nn.sigmoid
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.time_major=time_major

    def call(self, inputs, **kwargs):

        initial_state = tf.zeros((10, self.units))  # todo

        if not self.time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])

        lookback, _, _ = inputs.shape

        state = [initial_state, initial_state]

        outputs, states = [], []
        for time_step in range(lookback):
            _out, state = self.cell(inputs[time_step], state)

            outputs.append(_out)
            states.append(state)

        outputs = tf.stack(outputs)
        h_s = tf.stack([states[i][0] for i in range(lookback)])
        c_s = tf.stack([states[i][1] for i in range(lookback)])


        if not self.time_major:
            outputs = tf.transpose(outputs, [1, 0, 2])
            h_s = tf.transpose(h_s, [1, 0, 2])
            c_s = tf.transpose(c_s, [1, 0, 2])
            states = [h_s, c_s]
            last_output = outputs[:, -1]
        else:
            states = [h_s, c_s]
            last_output = outputs[-1]

        h = last_output

        if self.return_sequences:
            h = outputs

        if self.return_state:
            return h, states

        return h

    def cell(self, inputs, states):

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)

        x_i = K.dot(inputs, k_i)
        x_f = K.dot(inputs, k_f)
        x_c = K.dot(inputs, k_c)
        x_o = K.dot(inputs, k_o)

        if self.use_bias:
            b_i, b_f, b_c, b_o = array_ops.split(
                self.bias, num_or_size_splits=4, axis=0)
            x_i = K.bias_add(x_i, b_i)
            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        i = self.rec_activation(x_i + K.dot(h_tm1, self.rec_kernel[:, :self.units]))

        f = self.rec_activation(x_f + K.dot(h_tm1, self.rec_kernel[:, self.units:self.units * 2]))

        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.rec_kernel[:, self.units * 2:self.units * 3]))

        o = self.rec_activation(x_o + K.dot(h_tm1, self.rec_kernel[:, self.units * 3:]))

        h = o * self.activation(c)

        return h, [h, c]

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer=self.bias_initializer)

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer)

        self.rec_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer)

        self.built = True
        return


##############################################

seed_all(313)

lstm = LSTM(lstm_units, return_sequences=True)
h1 = lstm(inputs_tf)
h1_sum = tf.reduce_sum(h1)
print(K.eval(h1_sum))


##############################################

seed_all(313)
lstm = KLSTM(lstm_units,
             recurrent_activation="sigmoid",
             unit_forget_bias=False,
             return_sequences=True
            )
h2 = lstm(inputs_tf)
h2_sum = tf.reduce_sum(h2)
print(K.eval(h2_sum))

##############################################
# builing Model and training
#---------------------------------------------

# It is possible to use our vanilla LSTM as a layer in Keras Model.

##############################################

seed_all(313)
inp = Input(batch_shape=(10, lookback_steps, num_inputs))
lstm = LSTM(8)(inp)
out = Dense(1)(lstm)
model = Model(inputs=inp, outputs=out)
model.compile(loss='mse')

xx = np.random.random((100, lookback_steps, num_inputs))
y = np.random.random((100, 1))
h = model.fit(x=xx, y=y, batch_size=10, epochs=10)

##############################################

print(np.sum(h.history['loss']))

##############################################

# now compare the results by using original Keras LSTM i.e. KLSTM

seed_all(313)
inp = Input(batch_shape=(10, lookback_steps, num_inputs))
lstm = KLSTM(8,
             recurrent_activation="sigmoid",
             unit_forget_bias=False
             )(inp)
out = Dense(1)(lstm)
model = Model(inputs=inp, outputs=out)
model.compile(loss='mse')

xx = np.random.random((100, lookback_steps, num_inputs))
y = np.random.random((100, 1))
h = model.fit(x=xx, y=y, batch_size=10, epochs=10)
##############################################

print(np.sum(h.history['loss']))

##############################################

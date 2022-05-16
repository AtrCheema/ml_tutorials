"""
============================================
Implementing LSTM in numpy from scratch
============================================

The purpose of this notebook is to illustrate how to build an LSTM from scratch in numpy.

"""

import numpy as np
np.__version__


##############################################

import tensorflow as tf
tf.__version__

##############################################

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM as KLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Orthogonal, GlorotUniform, Zeros


assert tf.__version__ > "2.1", "results are not reproducible with Tensorflow below 2"

##############################################
# experiment setup

num_inputs = 3  # number of input features
lstm_units = 32
lookback_steps = 5  # also known as time_steps or sequence length
num_samples = 10  # length of x,y

##############################################
# in order to make the results comparable between tensorflow ansd numpy
# we use same weights for tensorflow inmplementation and numpy implementation

##############################################

k_init = GlorotUniform(seed=313)
k_vals = k_init(shape=(num_inputs, lstm_units*4))

rec_init = Orthogonal(seed=313)
rec_vals = rec_init(shape=(lstm_units, lstm_units*4))

b_init = Zeros()
b_vals = b_init(lstm_units*4)

weights = [k_vals, rec_vals, b_vals]

##############################################
# Keras version of LSTM
#---------------------------------------------

# check the results of forward pass of original LSTM of Keras

inp = Input(shape=(lookback_steps, num_inputs))
lstm_lyr = KLSTM(lstm_units)
out = lstm_lyr(inp)

lstm_lyr.set_weights(weights)

model = Model(inputs=inp, outputs=out)

xx = np.random.random((num_samples, lookback_steps, num_inputs))

lstm_out_tf = model.predict(x=xx)

##############################################
# numpy version of LSTM
#---------------------------------------------

class LSTMNP(object):
    """vanilla LSTM in pure numpy
    Only forward loop"""
    def __init__(self, units, return_sequences=False, time_major=False):
        self.units = units
        self.return_sequences = return_sequences
        self.time_major = time_major

        self.kernel = k_vals.numpy()
        self.rec_kernel = rec_vals.numpy()
        self.bias = b_vals.numpy()


    def __call__(self, inputs, initial_state=None):

        if not self.time_major:
            inputs = np.moveaxis(inputs, [0, 1], [1, 0])

        lookback_steps, bs, ins = inputs.shape

        if initial_state is None:
            h_state = np.zeros((bs, self.units))
            c_state = np.zeros((bs, self.units))
        else:
            assert len(initial_state) == 2
            h_state, c_state = initial_state


        h_states = []
        c_states = []

        for step in range(lookback_steps):

            h_state, c_state = self.cell(inputs[step], h_state, c_state)

            h_states.append(h_state)
            c_states.append(c_state)

        h_states = np.stack(h_states)
        c_states = np.stack(c_states)

        if not self.time_major:
            h_states = np.moveaxis(h_states, [0, 1], [1, 0])
            c_states = np.moveaxis(c_states, [0, 1], [1, 0])

        o = h_states[:, -1]
        if self.return_sequences:
            o = h_states

        return o

    def cell(self, xt, ht, ct):

        # input gate
        k_i = self.kernel[:, :self.units]
        rk_i = self.rec_kernel[:, :self.units]
        b_i = self.bias[:self.units]
        i_t = self.sigmoid(np.dot(xt, k_i) + np.dot(ht, rk_i) + b_i)

        # forget gate
        k_f = self.kernel[:, self.units:self.units * 2]
        rk_f = self.rec_kernel[:, self.units:self.units * 2]
        b_f = self.bias[self.units:self.units * 2]
        ft = self.sigmoid(np.dot(xt, k_f) + np.dot(ht, rk_f) + b_f)

        # candidate cell state
        k_c = self.kernel[:, self.units * 2:self.units * 3]
        rk_c = self.rec_kernel[:, self.units * 2:self.units * 3]
        b_c = self.bias[self.units * 2:self.units * 3]
        c_t = self.tanh(np.dot(xt, k_c) + np.dot(ht, rk_c) + b_c)

        # cell state
        ct = ft * ct + i_t * c_t

        # output gate
        k_o = self.kernel[:, self.units * 3:]
        rk_o = self.rec_kernel[:, self.units * 3:]
        b_o = self.bias[self.units * 3:]
        ot = self.sigmoid(np.dot(xt, k_o) + np.dot(ht, rk_o) + b_o)

        # hidden state
        ht = ot * self.tanh(ct)

        return ht, ct

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

##############################################

nplstm = LSTMNP(lstm_units)
lstm_out_np = nplstm(xx)

##############################################
# we can make sure that the results of numpy implementation and
# implementation of Tensorflow are exactly same

print(np.allclose(lstm_out_tf, lstm_out_np))

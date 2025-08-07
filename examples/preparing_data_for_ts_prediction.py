"""
============================================
Data Preparation for Time Series Prediction
============================================

This example demonstrates how to prepare data for time series prediction especially for
deep learning models/algorithms like LSTM/RNN.

"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

print("tf: ", tf.__version__)
print("np: ", np.__version__)

from utils import prepare_data
# %%
# Here we create a simple dataset with 2000 rows and 1 columns i.e. a
# univariate time series with no covariates.

rows = 2000
cols = 1
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()

# %%
# Below we print the first 10 rows, the shape of the dataset, and the last 10 rows to give an overview
# of the data structure.

print(data[0:10])  
print('\n {} \n'.format(data.shape))
print(data[-10:])

# %%
x, _y, y = prepare_data(data, num_inputs=1, num_outputs=1, lookback=4)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%
# Here we create a simple dataset with 2000 rows and 6 columns. Each column can represent
# a different feature or variable in the time series data. The dataset is filled with sequential
# integers for demonstration purposes.
rows = 2000
cols = 6
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()
print(data[0:10])  
print('\n {} \n'.format(data.shape))
print(data[-10:])


# %%
# multivariate time series with no covariates

x, _y, y = prepare_data(data, num_inputs=6, num_outputs=6, lookback=4)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%
x, _y, y = prepare_data(data, num_inputs=5, lookback=4)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%

x, _y, y = prepare_data(data, num_inputs=4, lookback=4)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%
# nowcasting vs forecasting
# --------------------------

x, _y, y = prepare_data(data, num_inputs=5, lookback=4, forecast_step=1)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%
# if we want to forecast multiple timesteps in future

x, _y, y = prepare_data(data, num_inputs=5, lookback=4, forecast_step=1, forecast_len=2)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%

x, _y, y = prepare_data(data, num_inputs=5, lookback=4, forecast_step=0, forecast_len=2)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%
x, _y, y = prepare_data(data, num_inputs=5, lookback=1, forecast_step=0)

# %%
# chaning input_steps
x, _y, y = prepare_data(data, num_inputs=5, lookback=4, input_steps=2)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]


# %%
# chaning output_steps
x, _y, y = prepare_data(data, num_inputs=5, lookback=4, output_steps=2)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

_y[0]

# %%

y[0]

# %%

x[1]

# %%

_y[1]

# %%

y[1]

# %%
# Handling missing values
# --------------------------
# missing values in the output

data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()
rng = np.random.default_rng(seed=313)  # for reproducibility
# create a random mask for the last column
mask = rng.integers(0, 2, size=data[:, -1].shape).astype(bool)
# introduce NaNs in the last column
data = data.astype(float)
data[mask, -1] = None

print(data[0:10])  
print('\n {} \n'.format(data.shape))
print(data[-10:])

# %%

x, _y, y = prepare_data(data, num_inputs=5, lookback=4)
print(x.shape, _y.shape, y.shape)


# %%

y[0]

# %%

y[1]

# %%

y[2]

# %%
y[3]

# %%
y[4], y[5], y[6]

# %%
# removing all examples with NaN in the output
nan_idx_y = np.isnan(y).any(axis=(1, 2))

non_nan_idx_y = np.invert(nan_idx_y)

x = x[non_nan_idx_y]
_y = _y[non_nan_idx_y]
y = y[non_nan_idx_y]

print(x.shape, _y.shape, y.shape)

# missing values in the input
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()
rng = np.random.default_rng(seed=313)  # for reproducibility
# put missing at random positions in the input data
mask = rng.integers(0, 50, size=data[:, :-1].shape).astype(bool)
data = data.astype(float)
data[:, :-1][~mask] = np.nan

print(data[0:10])  
print('\n {} \n'.format(data.shape))
print(data[-10:])

# %%
x, _y, y = prepare_data(data, num_inputs=5, lookback=5)
print(x.shape, _y.shape, y.shape)

x[-3]

# %%

x[-4]

# %%

y[-4]

# %%
# removing all examples with NaN in the input

nan_idx_x = np.isnan(x).any(axis=(1, 2))

non_nan_idx_x = np.invert(nan_idx_x)

x = x[non_nan_idx_x]
_y = _y[non_nan_idx_x]
y = y[non_nan_idx_x]

print(x.shape, _y.shape, y.shape)


# %%
# making batches
# --------------
lookback = 4
num_inputs = 5
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()
x, _y, y = prepare_data(data, num_inputs=num_inputs, lookback=lookback)
print(x.shape, _y.shape, y.shape)

# %%
inputs = Input(shape=(lookback, num_inputs))
lstm = LSTM(32)(inputs)
output = Dense(1)(lstm)
model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='mse')

# %%
model.fit(x, y, epochs=2, batch_size=128)

pred = model.predict(x)

# %%
# using generator
# ---------------
# In previous example, we had 1997 samples/examples, and each sample had shape (4, 5).
# This is small data and we can fit it in memory.
# But in real world, we may have large datasets with e.g. millions of samples/examples
# that cannot fit in memory.
# In such cases, we can use a data generator to load and preprocess the data in batches ourselves.

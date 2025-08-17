"""
============================================
Data Preparation for Time Series Prediction
============================================

This example demonstrates how to prepare data for time series prediction especially for
deep learning models/algorithms like LSTM/RNN.

"""

import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

from aqua_fetch import RainfallRunoff

print("tf: ", tf.__version__)
print("np: ", np.__version__)
print('pd: ', pd.__version__)

from utils import prepare_data, prepare_data_sample
# %%
# First we create a simple dataset with 2000 rows and 1 columns i.e. a
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
# Now we create a simple dataset with 2000 rows and 6 columns i.e. multivariate timeseries. Each column can represent
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
# If forecast_step is > 0, it means we want to predict in future. 
# It means we are predicting at t = t+1 which effectively means that we feed input data
# at timestep t and predict the target at timestep t+1.

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
# If forecast_step is 0, that means make prediction at t=0 which means we are 
# using the current input to predict the current output

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
# changing input_steps
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
# changing output_steps
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
# using known future inputs 
x, _y, y = prepare_data(data,
                        num_inputs=5,
                        lookback=4,
                        forecast_step=1,
                        forecast_len=4,
                        known_future_inputs=True)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

y[0]

# %%

x[1]

# %%

y[1]
# %%
# using known future inputs with forecast_step=2

x, _y, y = prepare_data(data, 
                        num_inputs=5, 
                        lookback=4,
                        forecast_len=4,
                        forecast_step=2,
                        input_steps=2, 
                        output_steps=2,
                        known_future_inputs=True)

print(x.shape, _y.shape, y.shape)

# %%

x[0]

# %%

y[0]

# %%

x[1]

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

cols = 6
rows = 200
lookback = 4
num_inputs = 5
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()

x0, _, y0 = prepare_data_sample(data, index=0, lookback=lookback, num_inputs=num_inputs)

x0

# %%

y0

# %%

x1, _, y1 = prepare_data_sample(data, index=1, lookback=lookback, num_inputs=num_inputs)

x1

# %%

y1


# %%

x4, _y4, y4 = prepare_data_sample(data, index=4, lookback=lookback, num_inputs=num_inputs)

x4

# %%

y4

# %%

def sample_generator(data:np.array, 
                     lookback, num_inputs, num_outputs=None, input_steps=1, forecast_step=0, forecast_len=1, known_future_inputs=False, output_steps=1):

    for i in range(len(data) - lookback * input_steps + 1 - forecast_step - forecast_len * output_steps):
        x, _, y = prepare_data_sample(data, index=i, lookback=lookback, 
                                        num_inputs=num_inputs, 
                                        num_outputs=num_outputs,
                                        input_steps=input_steps,
                                        forecast_step=forecast_step,
                                        forecast_len=forecast_len,
                                        known_future_inputs=known_future_inputs,
                                        output_steps=output_steps
                                        )

        # Skip samples with NaNs in x or y
        if np.isnan(x).any() or np.isnan(y).any():
            continue

        yield x, y

gen = sample_generator(data, lookback, num_inputs)

for idx, (x, y) in enumerate(gen):
    print(idx, x.shape, y.shape)

# %%
# Since we have drawn all the samples from generator and thus generator is exhausted
# we don't get anymore samples from it
for idx, (x, y) in enumerate(gen):
    print(idx, x.shape, y.shape)

# %%

output_signature = (
    tf.TensorSpec(shape=(4, 5), dtype=tf.float32),  # shape and dtype for x
    tf.TensorSpec(shape=(1, 1), dtype=tf.float32)   # shape and dtype for y
)

dataset = tf.data.Dataset.from_generator(
    sample_generator,
    args=(data, lookback, num_inputs),
    output_signature=output_signature
)

dataset

# %%

for idx, (x,y) in enumerate(dataset):
    print(idx, type(x), type(y), x.shape, y.shape)

# %%
# getting batches instead of single samples (x,y pairs) during iteration
dataset = tf.data.Dataset.from_generator(
    sample_generator,
    args=(data, lookback, num_inputs),
    output_signature=output_signature
)

batch_size = 32
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

dataset

# %%

for idx, (x,y) in enumerate(dataset):
    print(idx, type(x), type(y), x.shape, y.shape)

# %%

ds = RainfallRunoff('CAMELS_COL', verbosity=0)

static, dynamic = ds.fetch()

type(dynamic), len(dynamic)

# %%

dynamic['26247030'].shape

# %%
# get the total length of all DataFrames in dynamic

sum(df.shape[0] for df in dynamic.values())

# %%
# get the total length after dropping nan in last column

sum(df.dropna(subset=[df.columns[-1]]).shape[0] for df in dynamic.values())

# %%

def sample_generator(
        station_ids, 
        lookback:int, 
        num_inputs:int, 
        num_outputs=None, input_steps=1, forecast_step=0, forecast_len=1, known_future_inputs=False, output_steps=1):

    for stn in station_ids:

        stn = stn.decode() if isinstance(stn, bytes) else stn

        data = dynamic[stn].values

        for i in range(len(data) - lookback * input_steps + 1 - forecast_step - forecast_len * output_steps):
            x, _, y = prepare_data_sample(data, index=i, lookback=lookback, 
                                            num_inputs=num_inputs, 
                                            num_outputs=num_outputs,
                                            input_steps=input_steps,
                                            forecast_step=forecast_step,
                                            forecast_len=forecast_len,
                                            known_future_inputs=known_future_inputs,
                                            output_steps=output_steps
                                            )

            # Skip samples with NaNs in x or y
            if np.isnan(x).any() or np.isnan(y).any():
                continue

            yield x, y

lookback = 365
num_inputs = dynamic['26247030'].shape[1] - 1

output_signature = (
    tf.TensorSpec(shape=(lookback, num_inputs), dtype=tf.float32),  # shape and dtype for x
    tf.TensorSpec(shape=(1, 1), dtype=tf.float32)   # shape and dtype for y
)

dataset = tf.data.Dataset.from_generator(
    sample_generator,
    args=(list(dynamic.keys())[0:34], lookback, num_inputs),
    output_signature=output_signature
)

dataset

# %%
start = time.time()
for idx, (x,y) in enumerate(dataset):
    pass
print(round(time.time() - start, 2), 'seconds taken')
print("index of last sample: ", idx)
print(x.shape, y.shape)
# %%

dataset = tf.data.Dataset.from_generator(
    sample_generator,
    args=(list(dynamic.keys())[0:34], lookback, num_inputs),
    output_signature=output_signature
)

batch_size = 1024
dataset = dataset.take(1_000_000)  # Limit to 1 million samples
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

dataset

# %%
start = time.time()
for idx, (x,y) in enumerate(dataset):
    pass
print(round(time.time() - start, 2), 'seconds taken')
print("index of last batch: ", idx)
print(x.shape, y.shape)

# %%
# using tf.keras utility function which highly optimized

data = pd.concat([val for val in list(dynamic.values())[0:34]], axis=0)
print(data.shape)
dataset = tf.keras.utils.timeseries_dataset_from_array(
    data.iloc[:, 0:-1].values,
    targets=data.iloc[:, -1].values,
    sequence_length=lookback,
    batch_size=batch_size
)

dataset

# %%
start = time.time()
for idx, (x,y) in enumerate(dataset):
    pass
print(round(time.time() - start, 2), 'seconds taken')

print("index of last batch: ", idx)
print(x.shape, y.shape)

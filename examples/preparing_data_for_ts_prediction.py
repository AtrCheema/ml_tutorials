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
# Checking the first sample/example/data point
x[0]

# %%

_y[0]

# %%

y[0]

# %%
# Checking the second sample/example/data point
x[1]

# %%

_y[1]

# %%

y[1]

# %%
# Now we create another dataset with 2000 rows but with 6 columns i.e. multivariate timeseries. Each column can represent
# a different feature or variable in the time series data. The dataset is filled with sequential
# integers for demonstration purposes.

rows = 2000
cols = 6
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()
print(data[0:10])  
print('\n {} \n'.format(data.shape))
print(data[-10:])


# %%
# If this were a multivariate time series with no covariates then we would use 
# the same approach as before i.e. set the num_inputs equal to that of num_outputs.

x, _y, y = prepare_data(data, num_inputs=6, num_outputs=6, lookback=4)

print(x.shape, _y.shape, y.shape)

# %%
# Checking the first sample/example/data point
x[0]

# %%

_y[0]

# %%

y[0]

# %%
# Checking the second sample/example/data point
x[1]

# %%

_y[1]

# %%

y[1]

# %%
# However, if this were a multivariate time series with covariates, i.e. one
# timeseries column is our target variable and the others are input features,
# we would need to adjust the data preparation accordingly.

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
# Consider the case where number of input features/timeseries are 4 and output features/timeseries are 2.

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
# It reflects that we are predicting at timestep t = `t+1` which effectively means that we feed input data
# at timestep t and predict the target at timestep t+1.

x, _y, y = prepare_data(data, num_inputs=5, lookback=4, forecast_step=1)

print(x.shape, _y.shape, y.shape)

# %%
# First sample
x[0]

# %%

_y[0]

# %%

y[0]

# %%
# Second sample
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
# using input at current timestep to predict the output at current timestep.

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
# Consider the case where missing values are present in the output/target variable/feature

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
# Now we should remove all examples with NaN in the output. This will definitely
# reduce the number of samples.

nan_idx_y = np.isnan(y).any(axis=(1, 2))

non_nan_idx_y = np.invert(nan_idx_y)

x = x[non_nan_idx_y]
_y = _y[non_nan_idx_y]
y = y[non_nan_idx_y]

print(x.shape, _y.shape, y.shape)

# %%
# Now consider the case where missing values in the input features/variables as well

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
# We should definitely remove all examples with NaN in the input (x)

nan_idx_x = np.isnan(x).any(axis=(1, 2))

non_nan_idx_x = np.invert(nan_idx_x)

x = x[non_nan_idx_x]
_y = _y[non_nan_idx_x]
y = y[non_nan_idx_x]

print(x.shape, _y.shape, y.shape)

# %%
# making batches
# --------------
# A batch represents a group of samples/examples (x,y) pairs. The concept of batch
# is important in deep learning because neural networks are not training at once with
# all the data but are trained with batches i.e. we divide the whole data into batches
# then feed the a single batch to neural network , train with it and then feed the next
# batch.

lookback = 4
num_inputs = 5
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()
x, _y, y = prepare_data(data, num_inputs=num_inputs, lookback=lookback)
print(x.shape, _y.shape, y.shape)

# %%
# Consider the following example of training an LSTM with a data of of ~2000 samples.

inputs = Input(shape=(lookback, num_inputs))
lstm = LSTM(32)(inputs)
output = Dense(1)(lstm)
model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='mse')

# %%
model.fit(x, y, epochs=2, batch_size=128)
# %%
# We see that when we trained the model with whole data i.e. 1997 samples, there
# were 16 batches. This is because we set the batch size equal to 128.

pred = model.predict(x)

# %%
# using generator
# ---------------
# In previous example, we had 1997 samples/examples, and each sample had shape (4, 5).
# Our ``x`` contained all the samples/examples. This is a small data and we can fit it (all the samples) in memory.
# But in real world, we may have large datasets with e.g. millions of samples/examples
# that cannot fit in memory. This means we can not have x with millions of samples in memory especially
# when each sample is also large.
# In such cases, we can use a data generator to load and preprocess the data in batches ourselves.

cols = 6
rows = 200
lookback = 4
num_inputs = 5
data = np.arange(int(rows*cols)).reshape(-1,rows).transpose()

x0, _, y0 = prepare_data_sample(data, index=0, lookback=lookback, num_inputs=num_inputs)

x0

# %%
# The function prepare_data_sample returns a single sample/example/data point at a time
# using the `index` parameter to specify which sample to return.

y0

# %%
# So if we want to get the second sample/example/data point, we can call the function with index=1

x1, _, y1 = prepare_data_sample(data, index=1, lookback=lookback, num_inputs=num_inputs)

x1

# %%

y1


# %%
# Similarly, if we want to get the fifth sample/example/data point, we can call the function with index=4

x4, _, y4 = prepare_data_sample(data, index=4, lookback=lookback, num_inputs=num_inputs)

x4

# %%

y4

# %%
# Now we can create a generator function that yields samples from the dataset.

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
# Now we can prepare tensorflow Dataset using the generator.

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
# The `dataset` is a generator which returns a single sample (x,y) pair
# at each iteration

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
# Now when we iterate over `dataset`, we don't get a single sample/example
# (x,y) pair at each iteration but we get a batch of samples and the length/size
# of the batch is determined by the `batch_size` parameter.

for idx, (x,y) in enumerate(dataset):
    print(idx, type(x), type(y), x.shape, y.shape)

# %%
# Let's use a real world example. We get rainfall-runoff data for several hundred catchments/stations
# from Columbia.

ds = RainfallRunoff('CAMELS_COL', verbosity=0)

static, dynamic = ds.fetch()

type(dynamic), len(dynamic)

# %%
# dynamic is a dictionary with keys as station names and each value is a DataFrame.

dynamic['26247030'].shape

# %%
# get the total length of all DataFrames in dynamic

sum(df.shape[0] for df in dynamic.values())

# %%
# get the total length after dropping nan in last column

sum(df.dropna(subset=[df.columns[-1]]).shape[0] for df in dynamic.values())

# %%
# Now we make the sample_generator for given number of stations determined by `station_ids`

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
# Now we iterate over the `dataset` and measure the time taken
# to get all the samples from 34 stations. We chose 34 because it is a manageable number for our example.

start = time.time()
for idx, (x,y) in enumerate(dataset):
    pass
print(round(time.time() - start, 2), 'seconds taken')
print("index of last sample: ", idx)
print(x.shape, y.shape)
# %%
# getting batches instead of single samples (x,y pairs) during iteration

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
# Now when we iterate over `dataset`, we don't get a single sample/example
# (x,y) pair at each iteration but we get a batch of samples and the length/size
# of the batch is determined by the `batch_size` parameter.

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

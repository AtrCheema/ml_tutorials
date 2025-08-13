"""
=================
utility functions
=================
"""

from typing import Union, Tuple

import numpy as np


def prepare_data(
        data: np.ndarray,
        lookback: int,
        num_inputs: int = None,
        num_outputs: int = None,
        input_steps: int = 1,
        forecast_step: int = 0,
        forecast_len: int = 1,
        known_future_inputs: bool = False,
        output_steps: int = 1,
        mask: Union[int, float, np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    converts a numpy nd array into a supervised machine learning problem.

    Parameters
    ----------
        data :
            nd numpy array whose first dimension represents the number
            of examples and the second dimension represents the number of features.
            Some of those features will be used as inputs and some will be considered
            as outputs depending upon the values of `num_inputs` and `num_outputs`.
        lookback :
            number of previous steps/values to be used at one step.
        num_inputs :
            default None, number of input features in data. If None,
            it will be calculated as features-outputs. The input data will be all
            from start till num_outputs in second dimension.
        num_outputs :
            number of columns (from last) in data to be used as output.
            If None, it will be caculated as features-inputs.
        input_steps:
            strides/number of steps in input data
        forecast_step :
            must be greater than equal to 0, which t+ith value to
            use as target where i is the horizon. For time series prediction, we
            can say, which horizon to predict.
        forecast_len :
            number of horizons/future values to predict.
        known_future_inputs :
            Only useful if `forecast_len`>1. If True, this
            means, we know and use 'future inputs' while making predictions at t>0
        output_steps :
            step size in outputs. If =2, it means we want to predict
            every second value from the targets
        mask :
            If int, then the examples with these values in
            the output will be skipped. If array then it must be a boolean mask
            indicating which examples to include/exclude. The length of mask should
            be equal to the number of generated examples. The number of generated
            examples is difficult to prognose because it depend, upon lookback, input_steps,
            and forecast_step. Thus, it is better to provide an integer indicating
            which values in outputs are to be considered as invalid. Default is
            None, which indicates all the generated examples will be returned.

    Returns
    -------
        x : numpy array of shape (examples, lookback, ins) consisting of
            input examples
        prev_y : numpy array consisting of previous outputs
        y : numpy array consisting of target values of shape (examples, outs, forecast_len)

    Given following data consisting of input/output pairs

    +--------+--------+---------+---------+----------+
    | input1 | input2 | output1 | output2 | output 3 |
    +========+========+=========+=========+==========+
    |   1    |   11   |   21    |    31   |   41     |
    +--------+--------+---------+---------+----------+
    |   2    |   12   |   22    |    32   |   42     |
    +--------+--------+---------+---------+----------+
    |   3    |   13   |   23    |    33   |   43     |
    +--------+--------+---------+---------+----------+
    |   4    |   14   |   24    |    34   |   44     |
    +--------+--------+---------+---------+----------+
    |   5    |   15   |   25    |    35   |   45     |
    +--------+--------+---------+---------+----------+
    |   6    |   16   |   26    |    36   |   46     |
    +--------+--------+---------+---------+----------+
    |   7    |   17   |   27    |    37   |   47     |
    +--------+--------+---------+---------+----------+

    If we use following 2 time series as input

    +--------+--------+
    | input1 | input2 |
    +========+========+
    |  1     |  11    |
    +--------+--------+
    |     2  |  12    |
    +--------+--------+
    | 3      |  13    |
    +--------+--------+
    | 4      |  14    |
    +--------+--------+
    | 5      |  15    |
    +--------+--------+
    | 6      |  16    |
    +--------+--------+
    | 7      |  17    |
    +--------+--------+

    then  ``num_inputs`` =2, ``lookback`` =7, ``input_steps`` =1

    and if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |   27    |   37    |   47     |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =1,  ``forecast_step`` =0,

    if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    | 28      | 38      | 48       |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =1,  ``forecast_step`` =1,

    if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |  27     |  37     |  47      |
    +---------+---------+----------+
    |  28     |  38     |  48      |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =2,  horizon/forecast_step=0,

    if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |   28    |   38    |   48     |
    +---------+---------+----------+
    |   29    |   39    |   49     |
    +---------+---------+----------+
    |   30    |   40    |   50     |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =3,  ``forecast_step`` =1,

    if we want to predict

    +---------+
    | output2 |
    +=========+
    |   38    |
    +---------+
    |   39    |
    +---------+
    |   40    |
    +---------+

    then ``num_outputs`` =1, ``forecast_len`` =3, ``forecast_step`` =0

    if we predict

    +---------+
    | output2 |
    +=========+
    | 39      |
    +---------+

    then ``num_outputs`` =1, ``forecast_len`` =1, ``forecast_step`` =2

    if we predict

    +---------+
    | output2 |
    +=========+
    | 39      |
    +---------+
    | 40      |
    +---------+
    | 41      |
    +---------+

     then ``num_outputs`` =1, ``forecast_len`` =3, ``forecast_step`` =2

    If we use following two time series as input

    +--------+--------+
    |input1  | input2 |
    +========+========+
    |   1    |  11    |
    +--------+--------+
    |   3    |  13    |
    +--------+--------+
    |   5    |  15    |
    +--------+--------+
    |   7    |  17    |
    +--------+--------+

    then   ``num_inputs`` =2, ``lookback`` =4, ``input_steps`` =2

    If the input is

    +--------+--------+
    | input1 | input2 |
    +========+========+
    |    1   |  11    |
    +--------+--------+
    |    2   |  12    |
    +--------+--------+
    |    3   |  13    |
    +--------+--------+
    |    4   |  14    |
    +--------+--------+
    |    5   |  15    |
    +--------+--------+
    |    6   |  16    |
    +--------+--------+
    |   7    |  17    |
    +--------+--------+

    and target/output is

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |    25   |    35   |    45    |
    +---------+---------+----------+
    |    26   |    36   |    46    |
    +---------+---------+----------+
    |    27   |    37   |    47    |
    +---------+---------+----------+

    This means we make use of ``known future inputs``. This can be achieved using
    following configuration
    num_inputs=2, num_outputs=3, lookback=4, forecast_len=3, forecast_step=1, known_future_inputs=True

    The general shape of output/target/label is
    (examples, num_outputs, forecast_len)

    The general shape of inputs/x is
    (examples, lookback + forecast_len-1, ....num_inputs)


    Examples:
        >>> import numpy as np
        >>> from ai4water.utils.utils import prepare_data
        >>> num_examples = 50
        >>> dataframe = np.arange(int(num_examples*5)).reshape(-1, num_examples).transpose()
        >>> dataframe[0:10]
        array([[  0,  50, 100, 150, 200],
               [  1,  51, 101, 151, 201],
               [  2,  52, 102, 152, 202],
               [  3,  53, 103, 153, 203],
               [  4,  54, 104, 154, 204],
               [  5,  55, 105, 155, 205],
               [  6,  56, 106, 156, 206],
               [  7,  57, 107, 157, 207],
               [  8,  58, 108, 158, 208],
               [  9,  59, 109, 159, 209]])
        >>> x, prevy, y = prepare_data(dataframe, num_outputs=2, lookback=4,
        ...    input_steps=2, forecast_step=2, forecast_len=4)
        >>> x[0]
        array([[  0.,  50., 100.],
              [  2.,  52., 102.],
              [  4.,  54., 104.],
              [  6.,  56., 106.]], dtype=float32)
        >>> y[0]
        array([[158., 159., 160., 161.],
              [208., 209., 210., 211.]], dtype=float32)

        >>> x, prevy, y = prepare_data(dataframe, num_outputs=2, lookback=4,
        ...    forecast_len=3, known_future_inputs=True)
        >>> x[0]
        array([[  0,  50, 100],
               [  1,  51, 101],
               [  2,  52, 102],
               [  3,  53, 103],
               [  4,  54, 104],
               [  5,  55, 105],
               [  6,  56, 106]])       # (7, 3)
        >>> # it is important to note that although lookback=4 but x[0] has shape of 7
        >>> y[0]
        array([[154., 155., 156.],
               [204., 205., 206.]], dtype=float32)  # (2, 3)
    """
    if not isinstance(data, np.ndarray):
        if isinstance(data, pd.DataFrame):
            data = data.values
        else:
            raise TypeError(f"unknown data type for data {data.__class__.__name__}")

    if num_inputs is None and num_outputs is None:
        raise ValueError("""
Either of num_inputs or num_outputs must be provided.
""")

    features = data.shape[1]
    if num_outputs is None:
        num_outputs = features - num_inputs

    if num_inputs is None:
        num_inputs = features - num_outputs

#     assert num_inputs + num_outputs == features, f"""
# num_inputs {num_inputs} + num_outputs {num_outputs} != total features {features}"""

    if len(data) <= 1:
        raise ValueError(f"Can not create batches from data with shape {data.shape}")

    time_steps = lookback
    if known_future_inputs:
        lookback = lookback + forecast_len
        assert forecast_len > 1, f"""
            known_futre_inputs should be True only when making predictions at multiple 
            horizons i.e. when forecast length/number of horizons to predict is > 1.
            known_future_inputs: {known_future_inputs}
            forecast_len: {forecast_len}"""
        assert output_steps == input_steps == forecast_step, "different output_steps and input_steps with known_future_inputs are not supported yet"

    examples = len(data)

    x = []
    prev_y = []
    y = []

    for i in range(examples - lookback * input_steps + 1 - forecast_step - forecast_len * output_steps + 1):
        stx, enx = i, i + lookback * input_steps
        x_example = data[stx:enx:input_steps, 0:num_inputs]

        st, en = i, i + (lookback - 1) * input_steps
        y_data = data[st:en:input_steps, -num_outputs:]

        sty = (i + time_steps * input_steps) + forecast_step - input_steps
        eny = sty + forecast_len * output_steps
        if num_outputs == 0:
            target = np.array([]).reshape(forecast_len, 0)
        else:
            target = data[sty:eny:output_steps, -num_outputs:]

        x.append(np.array(x_example))
        prev_y.append(np.array(y_data))
        y.append(np.array(target))

    if len(x)<1:
        raise ValueError(f"""
        no examples generated from data of shape {data.shape} with lookback 
        {lookback} input_steps {input_steps} forecast_step {forecast_step} forecast_len {forecast_len}
""")
    x = np.stack(x)
    prev_y = np.array([np.array(i, dtype=np.float32) for i in prev_y], dtype=data.dtype)
    # transpose because we want labels to be of shape (examples, outs, forecast_len)
    y = np.array([np.array(i, dtype=np.float32).T for i in y], dtype=data.dtype)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            assert mask.ndim == 1
            assert len(x) == len(mask), f"Number of generated examples are {len(x)} " \
                                        f"but the length of mask is {len(mask)}"
        elif isinstance(mask, float) and np.isnan(mask):
            mask = np.invert(np.isnan(y))
            mask = np.array([all(i.reshape(-1,)) for i in mask])
        else:
            assert isinstance(mask, int), f"""
                    Invalid mask identifier given of type: {mask.__class__.__name__}"""
            mask = y != mask
            mask = np.array([all(i.reshape(-1,)) for i in mask])

        x = x[mask]
        prev_y = prev_y[mask]
        y = y[mask]

    return x, prev_y, y


def prepare_data_sample(
        data: np.ndarray,
        index: int,
        lookback: int,
        num_inputs: int = None,
        num_outputs: int = None,
        input_steps: int = 1,
        forecast_step: int = 0,
        forecast_len: int = 1,
        known_future_inputs: bool = False,
        output_steps: int = 1,
        mask: Union[int, float, np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    converts a numpy nd array into a supervised machine learning problem for a single sample.

    Parameters
    ----------
        data :
            nd numpy array whose first dimension represents the number
            of examples and the second dimension represents the number of features.
            Some of those features will be used as inputs and some will be considered
            as outputs depending upon the values of `num_inputs` and `num_outputs`.
        index :
            index of the sample to prepare. Must be within valid range based on
            lookback, forecast_step, and forecast_len parameters.
        lookback :
            number of previous steps/values to be used at one step.
        num_inputs :
            default None, number of input features in data. If None,
            it will be calculated as features-outputs. The input data will be all
            from start till num_outputs in second dimension.
        num_outputs :
            number of columns (from last) in data to be used as output.
            If None, it will be caculated as features-inputs.
        input_steps:
            strides/number of steps in input data
        forecast_step :
            must be greater than equal to 0, which t+ith value to
            use as target where i is the horizon. For time series prediction, we
            can say, which horizon to predict.
        forecast_len :
            number of horizons/future values to predict.
        known_future_inputs :
            Only useful if `forecast_len`>1. If True, this
            means, we know and use 'future inputs' while making predictions at t>0
        output_steps :
            step size in outputs. If =2, it means we want to predict
            every second value from the targets
        mask :
            If int or float, then the sample will be skipped if target values
            match the mask value. If None, no masking is applied.

    Returns
    -------
        x : numpy array of shape (lookback, ins) consisting of input example
        prev_y : numpy array consisting of previous outputs
        y : numpy array consisting of target values of shape (outs, forecast_len)

    Raises
    ------
        ValueError : if index is out of valid range or if data is insufficient
    """
    if not isinstance(data, np.ndarray):
        if hasattr(data, 'values'):  # pandas DataFrame
            data = data.values
        else:
            raise TypeError(f"unknown data type for data {data.__class__.__name__}")

    if num_inputs is None and num_outputs is None:
        raise ValueError("""
Either of num_inputs or num_outputs must be provided.
""")

    features = data.shape[1]
    if num_outputs is None:
        num_outputs = features - num_inputs

    if num_inputs is None:
        num_inputs = features - num_outputs

    if len(data) <= 1:
        raise ValueError(f"Can not create batches from data with shape {data.shape}")

    time_steps = lookback
    if known_future_inputs:
        lookback = lookback + forecast_len
        assert forecast_len > 1, f"""
            known_futre_inputs should be True only when making predictions at multiple 
            horizons i.e. when forecast length/number of horizons to predict is > 1.
            known_future_inputs: {known_future_inputs}
            forecast_len: {forecast_len}"""
        assert output_steps == input_steps == forecast_step, "different output_steps and input_steps with known_future_inputs are not supported yet"

    examples = len(data)
    
    # Check if index is within valid range
    max_valid_index = examples - lookback * input_steps + 1 - forecast_step - forecast_len * output_steps
    if index < 0 or index >= max_valid_index:
        raise ValueError(f"Index {index} is out of valid range [0, {max_valid_index-1}]")

    # Prepare single sample at given index
    i = index
    
    # Prepare input (x)
    stx, enx = i, i + lookback * input_steps
    x = data[stx:enx:input_steps, 0:num_inputs]

    # Prepare previous y
    st, en = i, i + (lookback - 1) * input_steps
    prev_y = data[st:en:input_steps, -num_outputs:]

    # Prepare target (y)
    sty = (i + time_steps * input_steps) + forecast_step - input_steps
    eny = sty + forecast_len * output_steps
    if num_outputs == 0:
        y = np.array([]).reshape(forecast_len, 0)
    else:
        target = data[sty:eny:output_steps, -num_outputs:]
        # transpose because we want labels to be of shape (outs, forecast_len)
        y = target.T

    # Apply mask if specified
    if mask is not None:
        if isinstance(mask, float) and np.isnan(mask):
            if np.any(np.isnan(y)):
                raise ValueError(f"Sample at index {index} contains NaN values in target")
        else:
            assert isinstance(mask, (int, float)), f"""
                    Invalid mask identifier given of type: {mask.__class__.__name__}"""
            if np.any(y == mask):
                raise ValueError(f"Sample at index {index} contains masked values in target")

    return x.astype(data.dtype), prev_y.astype(data.dtype), y.astype(data.dtype)
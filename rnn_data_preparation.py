import  numpy as np
__all__ = ['batch_generator']

def first_nan_from_end(ar):
    """ 
    This function finds index for first nan from the group which is present at the end of array.
    [np.nan, np.nan, 0,2,3,0,3, np.nan, np.nan, np.nan, np.nan] >> 7
    [np.nan, np.nan, 1,2,3,0, np.nan, np.nan, np.nan] >> 6
    [0,2,3,0,3] >> 5
    [np.nan, np.nan, 0,2,3,0,3] >> 7    
    """
    last_non_zero=0
    
    for idx, val in enumerate(ar[::-1]):
        if ~np.isnan(val): # val >= 0:
            last_non_zero = idx
            break
    return ar.shape[0] - last_non_zero


class batch_generator(object):
    """
    :param data: `ndarray`, input data.
    :param args: a dictionary containing values of parameters depending upon method used.
    :param method: str, default is 'many_to_one', if many_to_one, then following keys are expected in 
                   dictionary args.
            :lookback: `int`, sequence length, number of values LSTM will see at time `t` to make prediction at `t+1`.
            :in_features: `int`, number of columns in `data` starting from 0 to be considered as input
            :out_features: `int`, number of columns in `data` started from last to be considred as output/prediction.
            :trim_last_batch: bool, if True, last batch will be ignored if that contains samples less than `batch_size`.
            :norm: a dictionary which contains scaler object with which to normalize x and y data. We use separate scalers for x
                         and y data. Keys must be `x_scaler` and `y_scaler`.
            :batch_size:
            :step: step size in input data
            :min_ind: starting point from `data`
            :max_ind: end point from `data`
            :future_y_val: number of values to predict
    """
    
    def __init__(self, data, batch_size, args, method='many_to_one', verbose=True):
        
        self.data = data
        self.batch_size = batch_size
        self.args = args
        self.method=method
        self.verbose=verbose
        self.ignoriert_am_anfang=None
        self.ignoriert_am_ende = None
        self.no_of_batches = None
    

    def __len__(self):
        return self.args['min_ind'] - self.args['max_ind']
    
    def many_to_one(self):
    
        many_to_one_args = {'lookback': 'required',
                            'in_features': 'required',
                            'out_features': 'required',
                            'min_ind': 'required',
                            'max_ind': 'required',
                            'future_y_val': 'required',
                            'step': 1,
                            'norm': None,
                            'trim_last_batch':True}

        for k,v in many_to_one_args.items():
            if v=='required':
                if k not in self.args:
                    raise ValueError('for {} method, value of {} is required'.format(self.method, k))
                else:
                    many_to_one_args[k] = self.args[k]
            else:
                if k in self.args:
                    many_to_one_args[k] = self.args[k]

        lookback = many_to_one_args['lookback']
        in_features = many_to_one_args['in_features']
        out_features = many_to_one_args['out_features']
        min_ind = many_to_one_args['min_ind']
        max_ind = many_to_one_args['max_ind']
        future_y_val = many_to_one_args['future_y_val']
        step = many_to_one_args['step']
        norm = many_to_one_args['norm']
        trim_last_batch = many_to_one_args['trim_last_batch']

        # selecting the data of interest for x and y    
        X = self.data[min_ind:max_ind, 0:in_features]
        Y = self.data[min_ind:max_ind, -out_features:].reshape(-1,out_features)

        if norm is not None:
            x_scaler = norm['x_scaler']
            y_scaler = norm['y_scaler']
            X = x_scaler.fit_transform(X)
            Y = y_scaler.fit_transform(Y)

        # container for keeping x and y windows. A `windows` is here defined as one complete set of data at one timestep.
        x_wins = np.full((X.shape[0], lookback, in_features), np.nan, dtype=np.float32)
        y_wins = np.full((Y.shape[0], out_features), np.nan)

        # creating windows from X data
        st = lookback*step - step # starting point of sampling from data
        for j in range(st, X.shape[0]-lookback):
            en = j - lookback*step
            indices = np.arange(j, en, -step)
            ind = np.flip(indices)
            x_wins[j,:,:] = X[ind,:]

        # creating windows from Y data
        for i in range(0, Y.shape[0]-lookback):
            y_wins[i,:] = Y[i+lookback,:]



        """removing trailing nans"""
        first_nan_at_end = first_nan_from_end(y_wins[:,0])  # first nan in last part of data, start skipping from here
        y_wins = y_wins[0:first_nan_at_end,:]
        x_wins = x_wins[0:first_nan_at_end,:]

        """removing nans from start"""
        y_val = st-lookback + future_y_val
        if st>0:
            x_wins = x_wins[st:,:]
            y_wins = y_wins[y_val:,:]    

        if self.verbose:
            print("""shape of x data: {} \nshape of y data: {}""".format(x_wins.shape, y_wins.shape))

            print(""".\n{} values are skipped from start and {} values are skipped from end in output array"""
              .format(st, X.shape[0]-first_nan_at_end))
        self.ignoriert_am_anfang = st
        self.ignoriert_am_ende = X.shape[0]-first_nan_at_end

        pot_samples = x_wins.shape[0]

        if self.verbose:
            print('\npotential samples are {}'.format(pot_samples))

        residue = pot_samples % self.batch_size
        if self.verbose:
            print('\nresidue is {} '.format(residue))
        self.residue = residue

        samples = pot_samples - residue
        if self.verbose:
            print('\nActual samples are {}'.format(samples))

        interval = np.arange(0, samples + self.batch_size, self.batch_size)
        if self.verbose:
            print('\nPotential intervals: {}'.format(interval ))

        interval = np.append(interval, pot_samples)
        if self.verbose:
            print('\nActual interval: {} '.format(interval))

        if trim_last_batch:
            no_of_batches = len(interval)-2
        else:
            no_of_batches = len(interval) - 1 

        print('\nNumber of batches are {} '.format(no_of_batches))
        self.no_of_batches = no_of_batches

        # code for generator
        gen_i = 1
        while 1:

            for b in range(no_of_batches):
                st = interval[b]
                en = interval[b + 1]
                x_batch = x_wins[st:en, :, :]
                y_batch = y_wins[st:en]

                gen_i +=1

                yield x_batch, y_batch
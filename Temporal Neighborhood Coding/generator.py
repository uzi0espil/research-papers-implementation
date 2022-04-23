from tensorflow.keras.utils import Sequence
import numpy as np
from statsmodels.tsa.stattools import adfuller
import math
from typing import Optional, Union, Tuple, List

np.seterr(divide='ignore')


class TNCGenerator(Sequence):

    def __init__(self, 
                 X: np.ndarray,
                 batch_size: int,
                 n_samples: Union[int, Tuple[int, int], List[int]],
                 window_size: int, 
                 max_window: int = 3, 
                 shuffle: bool = True,
                 state: Optional[int] = None):
        super(Sequence, self).__init__()
        self.X = X  # shape: [trips, timesteps, n_features]
        self.T = X.shape[1]
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_window = max_window
        self.n_samples = n_samples
        self.n_samples_positive = n_samples if not isinstance(n_samples, (tuple, list)) else n_samples[0]
        self.n_samples_negative = n_samples if not isinstance(n_samples, (tuple, list)) else n_samples[1]
        self.state = state
        self.shuffle = shuffle

        # attributes
        self.epsilon = None
        self.delta = None
        
        self._right_padding = 0 if self.window_size % 2 == 0 else 1

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.X)

    def __getitem__(self, idx):
        minimum = idx * self.batch_size
        maximum = min(minimum + self.batch_size, self.X.shape[0])

        batch_x_t = []
        batch_x_close = []
        batch_x_distant = []
        for index in range(minimum, maximum):
            # get a random t within window_size.
            t = np.random.randint(2 * self.window_size, self.T - 2 * self.window_size)
            # get a random window centered by t
            batch_x_t.append(self.X[index][t - self.window_size // 2:t + self.window_size // 2 + self._right_padding])
            # get neighbors
            batch_x_close.append(self._get_neighbor(self.X[index], t))
            # get non neighbors
            batch_x_distant.append(self._get_non_neighbors(self.X[index], t))

        # X
        # batch_x_t => (batch_size * n_samples, window_size, n_features)
        batch_x_t = np.repeat(batch_x_t, max(self.n_samples_negative, self.n_samples_positive), axis=0)
        batch_x_close = np.asarray(batch_x_close).reshape((-1, self.window_size, self.X.shape[-1]))
        batch_x_distant = np.asarray(batch_x_distant).reshape((-1, self.window_size, self.X.shape[-1]))

        # y
        y_pu = np.ones((batch_x_distant.shape[0],))
        y_neighbors = np.ones((batch_x_close.shape[0],))
        y_non_neighbors = np.zeros((batch_x_distant.shape[0],))

        return (batch_x_t, batch_x_close, batch_x_distant), (y_pu, y_neighbors, y_non_neighbors), None

    def _get_neighbor(self, x, t):
        """Sample n_samples from neighborhood of point t in timeseries x where the neighborhood length is determined by ADF
        The operation is explained in section last paragraph in section 2.

        :param x: numpy array with shape [timeseries, n_features]
        :param t: a random point of time that is within x.
        :return: x points within the neighborhood of x[t], it has shape of [n_samples, window_size, features]
        """
        T = self.X.shape[-2]
        gap = self.window_size
        corr = []
        # check stationary gradually up to 3 size of window_size as stated in A.3
        for w_t in range(self.window_size, ((self.max_window + 1) * self.window_size), gap):
            try:
                p_val = 0
                for f in range(x.shape[-1]):  # for each feature, compute adfuller test.
                    with np.seterr(divide='ignore'):
                        p = adfuller(np.array(x[max(0, t - w_t):min(x.shape[-2], t + w_t), f].reshape(-1, )))[1]
                    p_val += 0.01 if math.isnan(p) else p  # add up the p-values of all tests unless one is not nan
                corr.append(p_val / x.shape[-1])  # take the mean of p-values
            except Exception as e:
                corr.append(0.6)
        non_stationary_indices = np.where(np.array(corr) >= 0.01)[0]

        # if all points until max_windows are stationary, then we take max_window.
        # Otherwise, we take the index of the first occurrence + 1.
        self.epsilon = self.max_window if len(non_stationary_indices) == 0 else non_stationary_indices[0] + 1
        self.delta = 5 * self.epsilon * self.window_size  # set delta for finding non_neighbors.

        # after determining the length of neighborhood based on ADF, time to randomly pick n_samples from the trip
        center = self.window_size // 2
        # add t to a Gaussian random multiplied by epsilon that is set by ADF and window_size.
        t_p = [int(t + np.random.randn() * self.epsilon * self.window_size) for _ in range(self.n_samples_positive)]
        t_p = [max(center + 1, min(t_pp, T - (center + self._right_padding))) for t_pp in t_p]
        x_p = np.stack([x[t_ind - center:t_ind + center + self._right_padding, :] for t_ind in t_p])
        return x_p

    def _get_non_neighbors(self, x, t):
        T = self.X.shape[-2]
        center = self.window_size // 2
        if t > T / 2:  # will get windows prior to t
            t_n = np.random.randint(center, max((t - self.delta + 1), center + 1), self.n_samples_negative)
        else:  # will get windows post to time t.
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size - 1)), (T - center), self.n_samples_negative)
        x_n = np.stack([x[t_ind - center:t_ind + center + self._right_padding, :] for t_ind in t_n])

        if len(x_n) == 0:  # if failed, then take anything within the window_size
            rand_t = np.random.randint(0, self.window_size // 5)  # five
            if t > T / 2:
                x_n = x[rand_t:rand_t + self.window_size, :].unsqueeze(0)
            else:
                x_n = x[T - rand_t - self.window_size:T - rand_t, :].unsqueeze(0)
        return x_n


def sliding_window(X: np.ndarray,
                   timesteps: int,
                   y: Optional[np.ndarray] = None,
                   shuffle: bool = False,
                   shift: int = 1,
                   dtype: Optional = np.float32):
    """
    Apply sliding window transformation where all data is loaded in memory using only numpy.

    :param X: features to apply the sliding window.
    :param timesteps: how many back steps in the past to look to.
    :param y: target(s) to apply the sliding window to. If None, then sliding window is only applied to X.
    :param shuffle: Shuffle X and y randomly.
    :param shift: shift between consecutive windows.
    :param dtype: type of the generated input and output.
    :return: transformed X and y.
    """
    timesteps_indices = timesteps

    if y is not None:
        window_size = timesteps - 1
        window_size = max(timesteps, window_size)
    else:
        window_size = timesteps

    min_index = 0
    max_index = len(X) - window_size + 1
    rows = np.arange(min_index, max_index, shift)

    if shuffle:
        np.random.shuffle(rows)

    samples = np.zeros((len(rows),
                        timesteps,
                        X.shape[-1]), dtype=dtype)

    targets = None
    if y is not None:
        targets = np.zeros((len(rows), y.shape[-1]), dtype=dtype)

    for j, row in enumerate(rows):
        # features
        indices = range(rows[j], rows[j] + timesteps_indices, 1)
        samples[j] = X[indices]

        # targets
        if y is not None:
            indices = range(rows[j] + timesteps_indices - 1,
                            rows[j] + timesteps_indices - 1,
                            1)
            targets[j] = y[indices]

    if y is not None:
        return samples, targets
    return samples

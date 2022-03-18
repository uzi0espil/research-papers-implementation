import tensorflow as tf
from typing import Optional
import numpy as np


def generator(X: np.ndarray,
              window_size: int,
              y: np.ndarray = None,
              batch_size: int = 32, 
              shift: int = 1,
              dilation: int = 1,
              cache: bool = True,
              shuffle: bool = True,
              num_parallel_calls: Optional[int] = None,
              prefetch: int = tf.data.experimental.AUTOTUNE,
              repeat: bool = True):
    
    def batch_timesteps(*args):
        # add current with future to y
        if len(args) > 1:
            y = tf.data.Dataset.zip((args[0].batch(window_size), args[1].batch(window_size)))
            return tf.data.Dataset.zip((args[0].batch(window_size), y))
        return args[0].batch(window_size)
    
    def features_targets(*args):
        X_window = args[0][:window_size]
        if len(args) > 1:
            y_window = args[1][:window_size]
            return X_window, y_window
        return X_window

    data = tf.data.Dataset.from_tensor_slices(X)
    if y is not None:
        y_data = tf.data.Dataset.from_tensor_slices(y)
        data = tf.data.Dataset.zip((data, y_data))
    data = data.window(window_size, shift=shift, stride=dilation, drop_remainder=True)
    data = data.flat_map(batch_timesteps)
    data = data.map(features_targets, num_parallel_calls=num_parallel_calls)

    if cache:
        data = data.cache()
    if shuffle:
        data = data.shuffle(X.shape[0] // window_size)
        
    data =  data.batch(batch_size)
    if repeat:
        data = data.repeat()
    return data.prefetch(prefetch)

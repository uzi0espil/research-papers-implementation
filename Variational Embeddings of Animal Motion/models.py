from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from typing import Optional, List
from typing_extensions import Literal


class Sampling(layers.Layer, ABC):

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GRUEncoder(tf.keras.Model, ABC):

    def __init__(self,
                 latent_size: int,
                 n_layers: List[int],
                 dropout: float = 0.,
                 bidirectional: bool = True,
                 noise_stddev: float = 0.8,
                 soft_plus: bool = False,
                 **kwargs):
        super(GRUEncoder, self).__init__(**kwargs)
        self.noise = None if noise_stddev is not None else layers.GaussianNoise(noise_stddev)
        self.hidden_layers = []
        for neurons in n_layers:
            gru = layers.GRU(neurons, dropout=dropout, return_state=True, return_sequences=True)
            if bidirectional:
                self.hidden_layers.append(layers.Bidirectional(gru))
            else:
                self.hidden_layers.append(gru)
        self.hidden_factor = (2 if bidirectional else 1) * len(n_layers)

        std_activation = 'softplus' if soft_plus else 'linear'
        self.mean = layers.Dense(latent_size)
        self.log_var = layers.Dense(latent_size, activation=std_activation)
        self.sampling = Sampling()

    def call(self, inputs, training=None, **kwargs):
        hiddens = []
        if self.noise:
            inputs = self.noise(inputs, **kwargs)

        for layer in self.hidden_layers:
            inputs, forward_h, backward_h = layer(inputs)
            hiddens.append(forward_h)
            hiddens.append(backward_h)

        hidden = tf.concat(hiddens, axis=1)

        mean = self.mean(hidden)
        log_var = self.log_var(hidden)
        if training:
            sampling = self.sampling((mean, log_var))
            return sampling, mean, log_var
        return mean, mean, log_var


class GRUDecoder(tf.keras.Model, ABC):

    def __init__(self,
                 n_features: int,
                 n_layers: List[int],
                 dropout: float = 0.,
                 bidirectional: bool = True,
                 **kwargs):
        super(GRUDecoder, self).__init__(**kwargs)
        self.hidden_layers = []
        self.bidirectional = bidirectional
        for i, neurons in enumerate(n_layers):
            gru = layers.GRU(neurons, dropout=dropout, return_sequences=True)
            if bidirectional:
                self.hidden_layers.append(layers.Bidirectional(gru))
            else:
                self.hidden_layers.append(gru)
        self.hidden_factor = (2 if bidirectional else 1) * len(n_layers)

        self.latent_to_hidden = layers.Dense(n_layers[0] * self.hidden_factor)  # linear activation ??
        self.hidden_to_output = layers.Dense(n_features)

    def call(self, inputs, **kwargs):
        ins, hidden = inputs

        hidden = self.latent_to_hidden(hidden, **kwargs)
        hidden = tf.split(hidden, self.hidden_factor, axis=1)

        for i, layer in enumerate(self.hidden_layers):
            initial_state = hidden[i:i + 1 * (2 if self.bidirectional else 1)]
            ins = layer(ins, initial_state=initial_state, **kwargs)

        return self.hidden_to_output(ins)


class KLAnnealWeight(tf.keras.callbacks.Callback, ABC):

    def __init__(self,
                 anneal_start: int = 2,
                 anneal_time: int = 4,
                 monitor: str = "val_loss",
                 early_exit: int = 10,
                 function: Literal["linear", "sigmoid"] = 'linear',
                 **kwargs):
        super(KLAnnealWeight, self).__init__(**kwargs)
        if function not in ["linear", "sigmoid"]:
            raise ValueError(f"Function {function} not supported.")

        self.anneal_start = K.cast_to_floatx(anneal_start)
        self.anneal_time = K.cast_to_floatx(anneal_time)
        self.monitor = monitor
        self.early_exit = early_exit
        self.function = function

        # attributes
        self.best = np.Inf
        self.count = 0
        self.kl_weight = 0.

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            K.set_value(self.model.kl_decay, 0.)
        if epoch >= self.anneal_start and self.kl_weight <= 0.99:
            if self.function.lower() == "linear":
                self.kl_weight = min(1, (epoch - self.anneal_start) / self.anneal_time)
            elif self.function == "sigmoid":
                self.kl_weight = float(1 / (1 + np.exp(-0.9 * (epoch - self.anneal_time))))
            K.set_value(self.model.kl_decay, self.kl_weight)
            if self.kl_weight > 0.99:
                print("kl anneal converged:", self.kl_weight)

    def on_epoch_end(self, epoch, logs=None):
        if self.kl_weight <= 0.99:
            return

        current_value = logs.get(self.monitor)
        if current_value <= self.best:
            self.count = 0
            self.best = current_value
        else:
            print(f"val_loss didn't improve from {self.best}")
            self.count += 1

        if self.count >= self.early_exit:
            self.model.stop_training = True

    def get_config(self):
        return dict(anneal_start=self.anneal_start,
                    anneal_time=self.anneal_time,
                    monitor=self.monitor,
                    early_exit=self.early_exit)


class VAME(tf.keras.Model, ABC):

    def __init__(self,
                 n_features: int,
                 n_clusters: int = 8,  # as default to KMeans in scikit-learn
                 n_layers_encoder: Optional[List[int]] = None,
                 n_layers_decoder: Optional[List[int]] = None,
                 latent_size: int = 10,
                 encoder_dropout: float = 0.,
                 decoder_dropout: float = 0.,
                 encoder_bidirectional: bool = True,
                 decoder_bidirectional: bool = True,
                 soft_plus: bool = False,
                 noise_stddev: Optional[float] = 0.8,
                 cluster_weight: float = 0.1,
                 kl_anneal_weight: KLAnnealWeight = KLAnnealWeight(),
                 kl_weight: float = 1.,
                 **kwargs):
        super(VAME, self).__init__(**kwargs)

        self.n_clusters = n_clusters
        self.cluster_weight = cluster_weight
        self.kl_anneal_callback = kl_anneal_weight
        self.kl_weight = kl_weight
        self.kl_decay = tf.Variable(0.)  # this should change by the KLAnnealWeight callback.

        if n_layers_decoder is None:
            n_layers_decoder = [64]

        if n_layers_encoder is None:
            n_layers_encoder = [64, 64]

        self.encoder = GRUEncoder(latent_size, n_layers_encoder, bidirectional=encoder_bidirectional,
                                  dropout=encoder_dropout, noise_stddev=noise_stddev, soft_plus=soft_plus)
        self.decoder = GRUDecoder(n_features, n_layers_decoder,
                                  bidirectional=decoder_bidirectional, dropout=decoder_dropout)
        self.decoder_future = GRUDecoder(n_features, n_layers_decoder,
                                         bidirectional=decoder_bidirectional, dropout=decoder_dropout)

        # metrics and losses
        self.reconstruction_loss = None
        self.reconstruction_metric = None
        self.reconstruction_future_metric = None
        self.loss_metric = None
        self.cluster_metric = None
        self.kl_metric = None

    def call(self, inputs, **kwargs):
        z, mu, logvar = self.encoder(inputs, training=True)

        ins = tf.expand_dims(z, axis=2)
        ins = tf.repeat(ins, tf.shape(inputs)[1], axis=2)
        ins = tf.transpose(ins, perm=[0, 2, 1])

        reconstructed_current = self.decoder((ins, z), training=True)
        reconstructed_future = self.decoder_future((ins, z), training=True)
        return reconstructed_current, reconstructed_future, z, mu, logvar

    def encode(self, inputs):
        mean, _, _ = self.encoder.predict(inputs)
        return mean

    @staticmethod
    def kullback_leibler_loss(mu, logvar):
        return -0.5 * tf.math.reduce_mean(1 + logvar - tf.square(mu) - tf.math.exp(logvar))

    def cluster_loss(self, h, batch_size):
        gram_matrix = (h @ tf.transpose(h)) / batch_size
        sv_2 = tf.linalg.svd(gram_matrix, compute_uv=False)
        sv = tf.math.square(sv_2[:self.n_clusters])
        loss = tf.math.reduce_sum(sv)
        return loss

    def compile(self, **kwargs):
        kwargs.update(dict(loss=None))
        super(VAME, self).compile(**kwargs)
        # initialize the losses and metrics
        self.reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="sum")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_metric = tf.keras.metrics.Mean(name="reconstruction_current")
        self.reconstruction_future_metric = tf.keras.metrics.Mean(name="reconstruction_future")
        self.kl_metric = tf.keras.metrics.Mean(name="kullback-divergence")
        self.cluster_metric = tf.keras.metrics.Mean(name="kmeans_loss")

    def train_step(self, data):
        X, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_current, y_future = y

        with tf.GradientTape() as tape:
            reconstructed_current, reconstructed_future, z, mu, logvar = self(X, training=True)

            reconstruction_loss = self.reconstruction_loss(y_current, reconstructed_current)
            reconstruction_future = self.reconstruction_loss(y_future, reconstructed_future)
            cluster_loss = self.cluster_loss(z, tf.cast(tf.shape(X)[0], dtype=K.floatx()))
            kl_loss = self.kullback_leibler_loss(mu, logvar)

            loss = (reconstruction_loss +
                    reconstruction_future +
                    (self.kl_weight * self.kl_decay * kl_loss) +
                    (self.cluster_weight * self.kl_decay * cluster_loss))

        all_trainable_variables = (self.encoder.trainable_variables +
                                   self.decoder.trainable_variables +
                                   self.decoder_future.trainable_variables)

        self.optimizer.minimize(loss, all_trainable_variables, tape=tape)

        self.loss_metric.update_state(loss)
        self.reconstruction_metric.update_state(reconstruction_loss)
        self.reconstruction_future_metric.update_state(reconstruction_future)
        self.kl_metric.update_state(self.kl_weight * self.kl_decay * kl_loss)
        self.cluster_metric.update_state(self.cluster_weight * self.kl_decay * cluster_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_current, y_future = y

        reconstructed_current, reconstructed_future, z, mu, logvar = self(X, training=False)

        reconstruction_loss = self.reconstruction_loss(y_current, reconstructed_current)
        reconstruction_future = self.reconstruction_loss(y_future, reconstructed_future)
        cluster_loss = self.cluster_loss(z, tf.cast(tf.shape(X)[0], dtype=K.floatx()))
        kl_loss = self.kullback_leibler_loss(mu, logvar)

        loss = (reconstruction_loss +
                reconstruction_future +
                (self.kl_weight * self.kl_decay * kl_loss) +
                (self.cluster_weight * self.kl_decay * cluster_loss))

        self.loss_metric.update_state(loss)
        self.reconstruction_metric.update_state(reconstruction_loss)
        self.reconstruction_future_metric.update_state(reconstruction_future)
        self.kl_metric.update_state(self.kl_weight * self.kl_decay * kl_loss)
        self.cluster_metric.update_state(self.cluster_weight * self.kl_decay * cluster_loss)

        return {m.name: m.result() for m in self.metrics}

    def fit(self, *args, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        kwargs.update(dict(callbacks=callbacks + [self.kl_anneal_callback]))
        return super(VAME, self).fit(*args, **kwargs)


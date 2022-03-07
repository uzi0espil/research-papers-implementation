from abc import ABC

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import tensorflow_addons as tfa


class TemporalBlock(tf.keras.layers.Layer, ABC):

    def __init__(self, n_filters, dilation_rate, kernel_size, strides, dropout, activation="relu",
                 weight_normalization=True, **kwargs):
        super(TemporalBlock, self).__init__(name="TemporalBlock", **kwargs)

        conv = tf.keras.layers.Conv1D(n_filters, kernel_size=kernel_size, strides=strides,
                                      dilation_rate=dilation_rate, padding="causal")

        self.conv = tfa.layers.WeightNormalization(conv) if weight_normalization else conv
        self.activation = tf.keras.layers.Activation(activation)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, **kwargs):
        out = self.conv(inputs, **kwargs)
        out = self.activation(out, **kwargs)
        return self.dropout(out, **kwargs)


class Encoder(tf.keras.layers.Layer, ABC):

    def __init__(self, latent_size, n_filters_per_block, kernel_size, strides, dropout, activation="relu", **kwargs):
        super(Encoder, self).__init__(name="encoder", **kwargs)
        self.activation = activation
        self.last_n_filters = n_filters_per_block[-1]

        self.temporal_blocks = []
        for i, n_filters in enumerate(n_filters_per_block):
            dilation_rate = 2 ** i
            self.temporal_blocks.append(TemporalBlock(n_filters, dilation_rate, kernel_size,
                                                      strides, dropout, activation=activation))
        self.flatten = tf.keras.layers.Flatten()
        self.z = None
        self.out = tf.keras.layers.Dense(latent_size)

    def build(self, input_shape):
        n_inputs = input_shape[-1]
        self.z = tf.keras.layers.Dense((self.last_n_filters * n_inputs // 2), activation=self.activation)
        super(Encoder, self).build(input_shape)

    def call(self, inputs, **kwargs):
        z = inputs
        for temporal_block in self.temporal_blocks:
            z = temporal_block(z, **kwargs)
        z = self.flatten(z)
        z = self.z(z)
        return self.out(z)


class Discriminator(tf.keras.layers.Layer, ABC):

    def __init__(self, hidden_neurons, dropout, activation="relu", **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.hidden_layers = []
        for neurons in hidden_neurons:
            self.hidden_layers.append(tf.keras.layers.Dense(neurons, activation=activation))
            if dropout > 0:
                self.hidden_layers.append(tf.keras.layers.Dropout(dropout))
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, **kwargs):
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z, **kwargs)
        return self.out(z, **kwargs)


class TNCModel(tf.keras.Model, ABC):

    def __init__(self,
                 i_shape, latent_size,
                 n_block_filters, kernel_size, unlabeled_weight,
                 disc_neurons, strides=1, dropout=0.2):
        super(TNCModel, self).__init__()
        self.latent_size = latent_size
        self.i_shape = i_shape
        self.unlabeled_weight = unlabeled_weight
        self.encoder = Encoder(latent_size, n_block_filters, kernel_size, strides, dropout)
        self.disc = Discriminator(disc_neurons, dropout)

        # additional metrics
        self.contrastive_loss = None
        self.p_acc = None
        self.n_acc = None

    def compile(self, *args, **kwargs):
        super(TNCModel, self).compile(*args, **kwargs)
        self.contrastive_loss = tf.keras.metrics.Mean(name="contrastive_loss")
        self.p_acc = tf.keras.metrics.Accuracy(name="positive_acc")
        self.n_acc = tf.keras.metrics.Accuracy(name="negative_acc")

    def call(self, inputs, **kwargs):
        """call can be removed in 2.7:
        https://github.com/keras-team/keras/commit/d9abcd788e5419560c4d8bd47b22ec387fe5c9c2
        """
        if isinstance(inputs, tuple):  # this should be removed by v2.7 keras
            inputs = inputs[0]
        return self.encoder(inputs, **kwargs)

    def _contrastive_loss(self, y_p, y_n, sample_weight=None):
        """
        Loss function based on Positive Unlabeled debaising technique where we treat the 
        negative samples as neutral samples where it can be seen as negative samples that 
        have positive sample traits (or might be drawn from the same distribution)
        
        The first term is about computing the cross-entropy of the positive samples.
        The second term is computing the cross-entropy of negative samples, once they are negative
        and once they are positive and weight them using `unlabeled_weight`.
        """
        y_true_p, y_pred_p = y_p
        y_true_n, y_pred_n = y_n

        p_loss = self.compiled_loss(y_p, y_pred_p, sample_weight=sample_weight)
        # compute once that they are negatives and once that they are positives
        n_loss = self.compiled_loss(y_n, y_pred_n, sample_weight=sample_weight)
        n_loss_u = self.compiled_loss(y_n, y_pred_p, sample_weight=sample_weight)

        return (p_loss + self.unlabeled_weight * n_loss_u + (1 - self.unlabeled_weight) * n_loss) / 2

    def train_step(self, data):
        X_all, y_all, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        X, X_p, X_n = X_all
        y_p, y_n = y_all

        with tf.GradientTape() as tape:
            t_embedding = self(X, training=True)
            p_embedding = self(X_p, training=True)
            n_embedding = self(X_n, training=True)

            y_pred_p = self.disc(tf.concat([t_embedding, p_embedding], axis=-1), training=True)
            y_pred_n = self.disc(tf.concat([t_embedding, n_embedding], axis=-1), training=True)

            loss = self._contrastive_loss((y_p, y_pred_p), (y_n, y_pred_n), sample_weight=sample_weight)

        all_trainable_variables = self.encoder.trainable_weights + self.disc.trainable_weights
        self.optimizer.minimize(loss, all_trainable_variables, tape=tape)

        # update contrastive loss
        self.contrastive_loss.update_state(loss)
        self.p_acc.update_state(y_p, y_pred_p > 0.5)
        self.n_acc.update_state(y_p, y_pred_n < 0.5)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X_all, y_all, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        X, X_p, X_n = X_all
        y_p, y_n = y_all

        # compute the embeddings
        t_embedding = self(X, training=False)
        p_embedding = self(X_p, training=False)
        n_embedding = self(X_n, training=False)

        y_pred_p = self.disc(tf.concat([t_embedding, p_embedding], axis=-1), training=False)
        y_pred_n = self.disc(tf.concat([t_embedding, n_embedding], axis=-1), training=False)

        # compute the loss
        loss = self._contrastive_loss((y_p, y_pred_p), (y_n, y_pred_n), sample_weight=sample_weight)

        # update validation metrics
        self.contrastive_loss.update_state(loss)
        self.p_acc.update_state(y_p, y_pred_p > 0.5)
        self.n_acc.update_state(y_p, y_pred_n < 0.5)

        return {m.name: m.result() for m in self.metrics}

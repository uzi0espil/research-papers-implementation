from abc import ABC
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List


def build_generator(image_dims, n_classes, codings_size, embedding_dims=50):
    width, height, channels = image_dims

    feature_map_width = width // 8
    feature_map_height = height // 8

    noise_inputs = tf.keras.layers.Input(shape=(codings_size,))
    noise_z = tf.keras.layers.Dense(feature_map_width * feature_map_height * 256)(noise_inputs)
    noise_z = tf.keras.layers.LeakyReLU(0.2)(noise_z)
    noise_z = tf.keras.layers.Reshape((feature_map_width, feature_map_height, 256))(noise_z)

    label_inputs = tf.keras.layers.Input(shape=(1,))
    label_z = tf.keras.layers.Embedding(input_dim=n_classes, output_dim=embedding_dims)(label_inputs)
    label_z = tf.keras.layers.Dense(feature_map_width * feature_map_height)(label_z)
    label_z = tf.keras.layers.Reshape((feature_map_width, feature_map_height, 1))(label_z)

    z = tf.keras.layers.Concatenate()([noise_z, label_z])
    z = tf.keras.layers.Conv2DTranspose(128, strides=2, kernel_size=5, padding="SAME")(z)
    z = tf.keras.layers.LeakyReLU(0.2)(z)

    z = tf.keras.layers.Conv2DTranspose(128, strides=2, kernel_size=5, padding="SAME")(z)
    z = tf.keras.layers.LeakyReLU(0.2)(z)

    z = tf.keras.layers.Conv2DTranspose(128, strides=2, kernel_size=5, padding="SAME")(z)
    z = tf.keras.layers.LeakyReLU(0.2)(z)

    out = tf.keras.layers.Conv2D(channels, kernel_size=3, activation="tanh", padding="SAME")(z)

    return tf.keras.Model(inputs=[noise_inputs, label_inputs], outputs=[out], name="generator")


class OneHotEncoder(tf.keras.layers.Layer):

    def __init__(self, n_classes, dtype=tf.float32, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.onehot_dtype = dtype
        self.reshape = tf.keras.layers.Reshape((n_classes,))

    def call(self, X_train, **kwargs):
        X_train = tf.cast(X_train, tf.int32)
        out = tf.one_hot(X_train, self.n_classes, dtype=self.onehot_dtype)
        return self.reshape(out)

    def get_config(self):
        return dict(n_classes=self.n_classes,
                    dtype=self.d_type)


def build_onehot_generator(image_dims, n_classes, codings_size):
    width, height, channels = image_dims

    feature_map_width = width // 4
    feature_map_height = height // 4

    noise_inputs = tf.keras.layers.Input(shape=(codings_size,))
    label_inputs = tf.keras.layers.Input(shape=(1,))
    label_z = OneHotEncoder(n_classes)(label_inputs)

    z = tf.keras.layers.Concatenate()([noise_inputs, label_z])
    z = tf.keras.layers.Dense(feature_map_width * feature_map_height * 64, activation="selu")(z)
    z = tf.keras.layers.Reshape((feature_map_width, feature_map_height, 64))(z)
    z = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="SAME", activation="selu")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="SAME", activation="selu")(z)
    out = tf.keras.layers.Conv2D(channels, kernel_size=3, activation="tanh", padding="SAME")(z)

    return tf.keras.Model(inputs=[noise_inputs, label_inputs], outputs=[out], name="generator")


def build_discriminator(image_dims, n_classes, embedding_dims=50):
    width, height, channels = image_dims

    image_inputs = tf.keras.layers.Input(shape=image_dims)
    label_inputs = tf.keras.layers.Input(shape=(1,))

    label_z = tf.keras.layers.Embedding(input_dim=n_classes, output_dim=embedding_dims)(label_inputs)
    label_z = tf.keras.layers.Dense(width * height)(label_z)
    label_z = tf.keras.layers.Reshape((width, height, 1))(label_z)

    z = tf.keras.layers.Concatenate()([image_inputs, label_z])

    z = tf.keras.layers.Conv2D(64, kernel_size=3, padding="SAME")(z)
    z = tf.keras.layers.LeakyReLU(0.2)(z)

    z = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="SAME")(z)
    z = tf.keras.layers.LeakyReLU(0.2)(z)

    z = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="SAME")(z)
    z = tf.keras.layers.LeakyReLU(0.2)(z)

    z = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding="SAME")(z)
    z = tf.keras.layers.LeakyReLU(0.2)(z)

    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dropout(0.4)(z)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(z)

    return tf.keras.Model(inputs=[image_inputs, label_inputs], outputs=[out], name="discriminator")


def build_onehot_discriminator(image_dims, n_classes):
    image_inputs = tf.keras.layers.Input(shape=image_dims)

    label_inputs = tf.keras.layers.Input(shape=(1,))
    label_z = OneHotEncoder(n_classes)(label_inputs)

    z = tf.keras.layers.Conv2D(32, strides=2, kernel_size=3)(image_inputs)
    z = tf.keras.layers.LeakyReLU(alpha=0.2)(z)
    z = tf.keras.layers.Dropout(0.4)(z)

    z = tf.keras.layers.Conv2D(64, strides=2, kernel_size=3)(z)
    z = tf.keras.layers.LeakyReLU(alpha=0.2)(z)
    z = tf.keras.layers.Dropout(0.4)(z)

    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Concatenate()([z, label_z])
    z = tf.keras.layers.Dense(512, activation="selu")(z)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(z)

    return tf.keras.Model(inputs=[image_inputs, label_inputs], outputs=[out], name="discriminator")


class ConditionalDCGAN(tf.keras.Model, ABC):

    def __init__(self, generator, discriminator, codings_size, n_classes, **kwargs):
        super(ConditionalDCGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.codings_size = codings_size
        self.n_classes = n_classes

        # define metrics
        self.d_loss, self.g_loss = None, None

    @property
    def metrics(self):
        return [self.d_loss, self.g_loss]

    def compile(self, g_optimizer, d_optimizer, *args, **kwargs):
        super(ConditionalDCGAN, self).compile(*args, optimizer=g_optimizer, **kwargs)
        self.discriminator.compile(*args, optimizer=d_optimizer, **kwargs)
        # initialize the metrics
        self.d_loss = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss = tf.keras.metrics.Mean(name="g_loss")

    def sampling(self, size: int = 1):
        samples = tf.random.normal((size, self.codings_size), )
        labels = tf.random.uniform((size, 1), minval=0, maxval=self.n_classes, dtype=tf.int32)
        labels = tf.cast(labels, tf.float32)
        return samples, labels

    def generate(self, size: int = 1, labels: Optional[Union[int, List[int]]] = None):
        """
        Generate new samples based on labels if given.

        :param size: number of samples to generate
        :param labels: the labels to generate, if None, then labels will be randomly chosen. In case of int, then all
        samples to be generated will be fromt that class. Otherwise, provide a list of labels equals to number of
        samples.
        :return: Newly generated samples.
        """
        samples, sampled_labels = self.sampling(size=size)

        if labels is not None:
            if isinstance(labels, int):
                labels = [labels] * size
            if size != len(labels):
                raise ValueError("Size and labels should have the same size")
            sampled_labels = tf.constant(labels, dtype=tf.float32)
            sampled_labels = tf.expand_dims(sampled_labels, axis=-1)

        images = self.generator((samples, sampled_labels))
        # inverse tanh transform:
        return (images + 1.) / 2.

    def call(self, X, training=True, **kwargs):
        _, X_cond = X
        y_gen = self.generator(X, training=training)
        return self.discriminator((y_gen, X_cond), training=training)

    def train_step(self, data):
        X_real, X_labels, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        batch_size = X_real.shape[0]

        # Phase 1
        X_sampled_noise, X_sampled_labels = self.sampling(size=batch_size)
        X_fake = self.generator((X_sampled_noise, X_sampled_labels))
        X_all_images = tf.concat([X_fake, X_real], axis=0)
        X_all_labels = tf.concat([X_sampled_labels, X_labels], axis=0)

        # Label smoothing as introduced by: Improved techniques for training GANs
        y_all = tf.constant([[0.]] * batch_size + [[0.9]] * batch_size)

        with tf.GradientTape() as tape:
            y_pred = self.discriminator((X_all_images, X_all_labels), training=True)
            d_loss = self.discriminator.compiled_loss(y_all, y_pred,
                                                      sample_weight=sample_weight,
                                                      regularization_losses=self.discriminator.losses)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Phase 2
        X_sampled_noise, X_sampled_labels = self.sampling(size=batch_size)
        y_fake = tf.constant([[1.]] * batch_size)

        with tf.GradientTape() as tape:
            y_preds = self((X_sampled_noise, X_sampled_labels), training=True)
            g_loss = self.compiled_loss(y_fake, y_preds,
                                        sample_weight=sample_weight,
                                        regularization_losses=self.losses)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss.update_state(d_loss)
        self.g_loss.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


class VisualizeSample(tf.keras.callbacks.Callback):

    def __init__(self, labels: List[str], n_images: int = 12, n_cols: int = 4):
        super(VisualizeSample, self).__init__()
        self.n_images = n_images
        self.n_cols = n_cols
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        _, labels = self.model.sampling(self.n_images)
        images = self.model.generate(size=self.n_images, labels=labels)
        n_cols = self.n_cols or len(images)
        n_rows = (len(images) - 1) // n_cols + 1
        if images.shape[-1] == 1:
            images = np.squeeze(images, axis=-1)
        plt.figure(figsize=(n_cols, n_rows))
        for index, image in enumerate(images):
            plt.subplot(n_rows, n_cols, index + 1)
            plt.gca().set_title(self.labels[int(labels.numpy()[index, 0])])
            plt.imshow(image, cmap="binary")
            plt.axis("off")
        plt.show()

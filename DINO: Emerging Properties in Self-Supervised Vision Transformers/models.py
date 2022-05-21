# +
from abc import ABC

import keras
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from keras import Model, layers
import keras_cv.layers.preprocessing as cv_layers
from typing import Union, Optional, Dict, List, Tuple
from tensorflow.python.keras.engine import data_adapter
from encoders import build_vit
from math import pi


class RandomApply(layers.Layer):

    def __init__(self, func, p=0.5, sampler="uniform", **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.p = p
        if sampler == "uniform":
            self.sampler = tf.random.uniform
        elif sampler == "normal":
            self.sampler = tf.random.normal
        else:
            raise ValueError("sampler can be either `uniform` or `normal`.")

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        randoms = tf.random.uniform((batch_size, 1, 1, 1), minval=0, maxval=1)
        augmented = self.func(inputs)
        return tf.where(randoms > 0.5, augmented, inputs)


class Augmentation(layers.Layer):

    def __init__(self,
                 resize: int,
                 crop_factor: Union[int, float],
                 gaussian_blur_p: float = 0.5,
                 solarization_p: Optional[float] = None,
                 interpolation: str = "bicubic",
                 n_crops: int = 1,
                 **kwargs):
        super(Augmentation, self).__init__(**kwargs)

        self.crop_factor = crop_factor
        self.n_crops = n_crops

        mean = [0.485, 0.456, 0.406]  # taken from original code. transferred to HxWxC
        var = np.square([0.229, 0.224, 0.225])  # taken from original code, in original they are std, keras needs var.

        self.rescale = layers.Rescaling(1./255)
        self.normalize = layers.Normalization(axis=-1, mean=mean, variance=var)

        # requires randomSampler of 0.8
        jitter = cv_layers.RandomColorJitter(value_range=(0, 255), brightness_factor=0.4, contrast_factor=0.4,
                                             saturation_factor=0.2, hue_factor=0.1)
        self.random_jitter = RandomApply(jitter, p=0.8)

        # requires randomSample of 0.2
        gray_scale = cv_layers.Grayscale(output_channels=3)
        self.random_gray_scale = RandomApply(gray_scale, p=0.2)

        gaussian_blur = cv_layers.RandomGaussianBlur(kernel_size=3, factor=(0.1, 2.))
        self.random_gaussian_blur = RandomApply(gaussian_blur, p=gaussian_blur_p)

        # 128 is taken from PIL ImageOps.Solarization, In DINO paper, they invert completely.
        self.to_solarize = solarization_p is not None
        solarization = cv_layers.Solarization(value_range=(0, 255), threshold_factor=(128, 255))
        self.random_solarization = RandomApply(solarization, p=solarization_p)

        self.flip = layers.RandomFlip(mode="horizontal")  # by default apply 0.5 probability.

        self.crop = None
        self.resize = layers.Resizing(height=resize, width=resize, interpolation=interpolation)

    def build(self, input_shape):
        w, h = input_shape[1], input_shape[0]
        crop_w, crop_h = self.get_params(w, h)
        self.crop = layers.RandomCrop(height=crop_w, width=crop_h)

    def get_params(self, width, height, ratio=1.):
        """In the original paper, the local crop is random taken from scale=(0.4, 1.0) and ratio=(3/4,4/3)
        I took the mean of the scale and ratio."""
        if isinstance(self.crop_factor, int):
            return self.crop_factor

        area = width * height
        target_area = area * self.crop_factor

        w = int(round(np.sqrt(target_area * ratio)))
        h = int(round(np.sqrt(target_area / ratio)))

        return w, h

    def call(self, inputs, *args, **kwargs):
        out = []
        for _ in range(self.n_crops):
            z = self.crop(inputs, **kwargs)
            z = self.resize(z, **kwargs)
            z = self.flip(z, **kwargs)
            z = self.random_jitter(z, **kwargs)
            z = self.random_gray_scale(z, **kwargs)
            z = self.random_gaussian_blur(z, **kwargs)
            if self.to_solarize:
                z = self.random_solarization(z, **kwargs)
            z = self.rescale(z, **kwargs)
            z = self.normalize(z, **kwargs)
            out.append(z)
        return tf.concat(out, axis=0)


def build_dino(output_shape: int,
               encoder: Union[Dict, keras.Model],
               neurons_per_hidden: Union[Tuple, List] = (2048, 2048, 256),
               activation: str = "gelu",
               norm_last_layer: bool = True):
    if isinstance(encoder, dict):
        encoder = build_vit(**encoder)

    dino_head = tf.keras.Sequential(name="DINO-head")
    for i, neuron in enumerate(neurons_per_hidden):
        dino_head.add(layers.Dense(neuron, activation=activation, name=f"representative-layer-{i}"))

    # Apply Euclidean (L2) normalization on the output of MLP
    dino_head.add(layers.Lambda(lambda x: tf.linalg.normalize(x, ord="euclidean", axis=-1)[0], name="L2-normalize"))

    # Apply Weight Normalization on the last layer.
    output_layer = layers.Dense(output_shape, activation="linear", use_bias=False, name="Output")
    if norm_last_layer:
        output_layer = WeightNormalization(output_layer, name="Normalized-output")
    dino_head.add(output_layer)

    # dino model
    return keras.Sequential([encoder, dino_head])


class UpdateTeacherCallback(keras.callbacks.Callback):
    """As indicated in the paper in section, the teacher is best updated at the end of the epoch and unlike student
    that is updated at every batch. This callback class is used to update the teacher at the very end of the epoch."""

    def __init__(self, decay_steps, initial_momentum=0.996, final_momentum=1., **kwargs):
        super(UpdateTeacherCallback, self).__init__(**kwargs)
        if not hasattr(self.model, "teacher") or not hasattr(self.model, "student"):
            raise ValueError("The used model has no `teacher` network within, this callback is used along side "
                             "self-distillation models.")
        self.decay_steps = decay_steps
        self.initial_momentum = 0.996
        self.final_momentum = 1.0

        self.teacher_momentum = self.initial_momentum

        # private variables
        self._epochs = 0
        self._n_iters = 0
        self._n_batches = 0

    def on_epoch_end(self, epoch, logs=None):
        for i, (param_s, param_t) in enumerate(zip(self.model.student.trainable_weights,
                                                   self.model.teacher.trainable_weights)):
            self.model.teacher.trainable_weights[i].assign((param_t * self.teacher_momentum) +
                                                           (1 - self.teacher_momentum) * param_s)
        self._n_iters = self._n_batches

    def on_epoch_begin(self, epoch, logs=None):
        self._epochs += 1

    def on_train_batch_begin(self, batch, logs=None):
        self._n_batches = batch + 1

    def on_train_batch_end(self, batch, logs=None):
        """Compute momentum for the next batch"""
        step = (self._n_iters * self._epochs) + self._n_batches + 1  # +1 for the next batch
        self.teacher_momentum = self.final_momentum + 0.5 * (self.initial_momentum - self.final_momentum) * (
                1 + tf.math.cos(pi * step / self.decay_steps)
        )


class DINO(Model, ABC):

    def __init__(self, encoder, momentum_decay_steps, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9,
                 teacher_update_momentum=0.996, n_local_views=8, *args, **kwargs):
        super(DINO, self).__init__(*args, **kwargs)
        self.student = encoder
        self.teacher = encoder
        self.teacher_update_momentum = teacher_update_momentum
        self.momentum_decay_steps = momentum_decay_steps

        self.student_augment = Augmentation(96, 0.225, n_crops=n_local_views)
        self.teacher_augment1 = Augmentation(224, 0.7, gaussian_blur_p=1.0, solarization_p=None)
        self.teacher_augment2 = Augmentation(224, 0.7, gaussian_blur_p=0.1, solarization_p=0.2)

        # teacher and student start with the same state
        self.teacher.set_weights(self.student.get_weights)

        # set the temperature
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # views
        self.n_local_views = n_local_views

        # set the center
        self.center_momentum = center_momentum
        self.center = tf.zeros((self.student.output,))

        # private variables
        self._n_global_views = 2
        self._total_views = self._n_global_views + self.n_local_views
        # this mask is to mask the global images computed by students and compute the images of different views
        self._loss_mask = tf.ones((self._n_global_images, self._total_views))
        self._loss_mask = tf.linalg.set_diag(self._loss_mask, diagonal=tf.zeros((self._n_global_views,)))
        self._n_terms = (self.n_local_views * self._n_global_views) - self._n_global_views

        # losses
        self.loss_metric = None

    def compile(self, **kwargs):
        kwargs = kwargs.pop("loss")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        super(DINO, self).compile(**kwargs)

    def self_distillation_loss(self, student_out, teacher_out):
        """Apply cross-entropy for each image of global view with all images of the local views
        except for the equal views, then the losses of all local images against the global images are summed
        and normalized.

        :param student_out: The output of the student's network, it consists of embeddings ot all global and local views
        :param teacher_out: The output of the teacher's network, it consists of embeddings of two global views.
        """
        # apply softmax for student output, the log is for computing the cross-entropy.
        student_out = tf.math.log_softmax(student_out / self.student_temp, dim=-1)

        # teacher centering and sharpening
        teacher_out = tf.math.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)

        teacher_out = tf.reshape(teacher_out, (self._n_global_views, -1, self.dim_out))
        teacher_out = tf.transpose(teacher_out, perm=(1, 0, 2))  # batch_size, n_global, out_dim
        student_out = tf.reshape(student_out, (self._total_views, -1, self.dim_out))
        student_out = tf.transpose(student_out, prem=(1, 2, 0))  # batch_size, out_dim, n_total

        losses = tf.matmul(-teacher_out, student_out)
        losses *= self._loss_mask

        return losses.sum() / self._n_terms

    def train_step(self, data):
        X, _, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        teacher_X1 = self.teacher_augment1(X, training=True)
        teacher_X2 = self.teacher_augment2(X, training=True)
        teacher_X = tf.concat([teacher_X1, teacher_X2], axis=0)  # [2 x batch_size, h, w, c]
        student_X = self.student_augment(X, training=True)  # [n_crops x batch_size, h, w, c]
        with tf.GradientTape() as tape:
            teacher_out = self.teacher(teacher_X, training=True)
            # Both global and local views are predicted by the student network.
            student_out_g = self.student(teacher_X, training=True)
            student_out_l = self.student(student_X, training=True)
            student_out = tf.concat([student_out_g, student_out_l], axis=0)  # [n_total_crops x batch_size, h, w ,c]

            loss = self.self_distillation_loss(student_out, teacher_out)

        # train only the student backward, the teacher is updated by callback.
        self.optimizer.minimize(loss, self.student.trainable_weights, tape=tape)

        # update center, it is never considered in the backpropagation.
        self.update_center(teacher_out)

        # set loss
        self.loss_metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def update_center(self, teacher_output):
        """Update center used for teacher output."""
        # by default we have to global images.
        batch_center = tf.math.reduce_sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / tf.shape(teacher_output)[0]

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def fit(self, *args, **kwargs):
        """Override fit function to force set UpdateTeacherCallback callback"""
        callbacks = kwargs.get("callbacks", [])
        epochs = kwargs.get("epochs", 1)
        callbacks.append(UpdateTeacherCallback(self.momentum_decay_steps, initial_momentum=self.teacher_update_momentum))
        kwargs.update(callbacks=callbacks)
        return super(DINO, self).fit(*args, **kwargs)


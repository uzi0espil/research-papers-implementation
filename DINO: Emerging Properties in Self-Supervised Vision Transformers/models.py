# +
from abc import ABC

import numpy as np
import tensorflow as tf
from keras import Model, layers
import keras_cv.layers.preprocessing as cv_layers
from typing import Union, Optional


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
                 **kwargs):
        super(Augmentation, self).__init__(**kwargs)

        self.crop_factor = crop_factor

        mean = [0.485, 0.456, 0.406]  # taken from original code.
        var = np.square([0.229, 0.224, 0.225])  # taken from original code, in original they are std, keras needs var.

        self.normalize = layers.Normalization(mean=mean, variance=var)

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
        z = self.crop(inputs, **kwargs)
        z = self.resize(z, **kwargs)
        z = self.flip(z, **kwargs)
        z = self.random_jitter(z, **kwargs)
        z = self.random_gray_scale(z, **kwargs)
        z = self.random_gaussian_blur(z, **kwargs)
        if self.to_solarize:
            z = self.random_solarization(z, **kwargs)
        return self.normalize(z, **kwargs)


class DINO(Model, ABC):

    def __init__(self, encoder, *args, **kwargs):
        super(DINO, self).__init__(*args, **kwargs)
        self.student = encoder
        self.teacher = encoder

        self.student_augment = Augmentation(96, 0.225)
        self.teacher_augment1 = Augmentation(224, 0.7, gaussian_blur_p=1.0, solarization_p=None)
        self.teacher_augment2 = Augmentation(224, 0.7, gaussian_blur_p=0.1, solarization_p=0.2)

    def self_distillation_loss(self):
        pass

    def train_step(self, data):
        pass


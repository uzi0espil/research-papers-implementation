import tensorflow as tf
from math import pi


class LinearCosineScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """LR scheduler that is first warms up with linear increase from start_warmup_value to base_value, then
    return LR based on Cosine scheduler that falls between base_value and final_value.

    :param base_value: is the first value to start for the cosine scheduler.
    :param final_value: is the final value to end the cosine scheduler with.
    :param n_iters: the total number of steps. Moreover, epochs * n_iters_per_epoch.
    :param warmup_steps: The number of steps to move from linear to cosine scheduler.
    :param start_warmup_value: the initial value to start the scheduler with.
    """

    def __init__(self, base_value, final_value, n_iters, warmup_steps=0, start_warmup_value=0):
        self.base_value = tf.constant(base_value)
        self.final_value = tf.constant(final_value)
        self.n_iters = n_iters

        self.warmup_steps = warmup_steps
        self.start_warmup_value = start_warmup_value

        self.decay_steps = tf.constant(n_iters - warmup_steps)
        self._current_lr = base_value if warmup_steps == 0 else start_warmup_value

    def __call__(self, step):
        # linear increase from start_warmup_value to base_value
        if self.warmup_steps > 0 and step < self.warmup_steps:
            if step == 0:
                return self._current_lr
            self._current_lr = self._current_lr + (self.base_value / self.warmup_steps)
        # cosine increase from base_value to final_value
        else:
            step = step - self.warmup_steps
            self._current_lr = self.final_value + 0.5 * (self.base_value - self.final_value) * (
                    1 + tf.math.cos(pi * step / self.decay_steps)
            )
        return self._current_lr

    def get_config(self):
        config = self.get_config()
        config.update(base_value=self.base_value, final_value=self.final_value, n_iters=self.n_iters,
                      warmup_steps=self.warmup_steps, start_warmup_value=self.start_warmup_value)
        return config

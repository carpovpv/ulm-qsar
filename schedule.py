import tensorflow as tf 

class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Native GPU-compiled Noam warmup + inverse square-root decay schedule."""

    def __init__(self, warmup_steps: int, lr_factor: float):
        super().__init__()
        self.warmup_steps = float(warmup_steps)
        self.lr_factor = float(lr_factor)

    def __call__(self, step):
        step = tf.cast(step + 1, tf.float32)
        arg1 = step / self.warmup_steps
        arg2 = 1.0
        return self.lr_factor * tf.minimum(arg1, arg2) / tf.maximum(step, self.warmup_steps)

    def get_config(self):
        return {"warmup_steps": self.warmup_steps, "lr_factor": self.lr_factor}



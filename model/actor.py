import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

class Actor(tf.keras.Model):

    def __init__(self):
        super().__init__()
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.dense1 = Dense(256, activation="relu")
        self.dense2 = Dense(256, activation="relu")
        self.dense3 = Dense(1, activation="tanh", kernel_initializer=last_init)

    def params_from_config(self, config):
        self.upper_bound = config["upper_bound"]
        self.lower_bound = config["lower_bound"]

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

    def get_action(self, state, noise=None):
        sampled_action = tf.squeeze(self.call(state))
        if noise is None:
            noise = 0
        else:
            noise = noise()
        return tf.squeeze(tf.clip_by_value(sampled_action+noise, self.lower_bound, self.upper_bound))

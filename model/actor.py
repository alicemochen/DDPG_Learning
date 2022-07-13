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

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

    def get_action(self, state):
        sampled_action = tf.squeeze(self.call(state))
        noise = np.random.randn()
        return sampled_action+noise

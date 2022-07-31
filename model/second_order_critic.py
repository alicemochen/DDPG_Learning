import tensorflow as tf
from tensorflow.keras.layers import *


class SecondOrderCritic(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = Dense(16, activation="relu")
        self.dense2 = Dense(32, activation="relu")
        self.dense3 = Dense(1, activation="relu")
        self.concat = Concatenate()

        self.dense6 = Dense(1, activation="relu")

    def call(self, inputs):
        state_input, action_input = inputs[0], inputs[1]
        concat = self.concat([state_input, action_input])
        out = self.dense1(concat)
        out = self.dense2(out)
        return self.dense6(out)
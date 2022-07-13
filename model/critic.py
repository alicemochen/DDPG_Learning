import tensorflow as tf
from tensorflow.keras.layers import *

class Critic(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = Dense(16, activation="relu")
        self.dense2 = Dense(32, activation="relu")

        self.dense3 = Dense(32, activation="relu")
        self.concat = Concatenate()
        self.dense4 = Dense(256, activation="relu")
        self.dense5 = Dense(256, activation="relu")
        self.dense6 = Dense(1)

    def call(self, inputs):
        state_input, action_input = inputs[0], inputs[1]
        state_out = self.dense1(state_input)
        state_out = self.dense2(state_out)
        action_out = self.dense3(action_input)
        concat = self.concat([state_out, action_out])
        out = self.dense4(concat)
        out = self.dense5(out)
        return self.dense6(out)
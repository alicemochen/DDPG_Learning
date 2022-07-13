import numpy as np
import tensorflow as tf

class Buffer:

    def __init__(self, config, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, config['state_features_num']))
        self.action_buffer = np.zeros((self.buffer_capacity, config['action_features_num']))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, config['state_features_num']))


    def record(self, observation):
        idx = self.buffer_counter % self.buffer_capacity
        self.state_buffer[idx] = observation[0]
        self.action_buffer[idx] = observation[1]
        self.reward_buffer[idx] = observation[2]
        self.next_state_buffer[idx] = observation[3]

        self.buffer_counter += 1

    def replay_sample(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch


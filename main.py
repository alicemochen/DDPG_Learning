import yaml
import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from model.actor import Actor
from model.buffer import Buffer
from model.critic import Critic


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def train(state_batch, action_batch, reward_batch, next_state_batch, target_actor, actor, target_critic, critic, tau, gamma):
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic(
            [next_state_batch, target_actions], training=True
        )
        critic_value = critic([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor(state_batch, training=True)
        critic_value = critic([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor.trainable_variables)
    )

    update_target(target_actor.variables, actor.variables, tau)
    update_target(target_critic.variables, critic.variables, tau)



if __name__ == '__main__':
    # 2 copies of actors and critics
    actor = Actor()
    critic = Critic()
    target_actor = Actor()
    target_critic = Critic()
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    problem = "Pendulum-v1"
    env = gym.make(problem)

    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    with open("pendulum_v1.yaml", "r") as f:
        config = yaml.safe_load(f)
    actor_optimizer = tf.keras.optimizers.Adam(config["actor_lr"])
    critic_optimizer = tf.keras.optimizers.Adam(config["critic_lr"])
    buffer = Buffer(config)

    # Takes about 4 min to train
    for ep in range(config["total_episodes"]):

        prev_state = env.reset()
        episodic_reward = 0
        i = 0
        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = actor.get_action(tf_prev_state)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step([action])
            buffer.record((prev_state, action, reward, state))
            state_batch, action_batch, reward_batch, next_state_batch = buffer.replay_sample()
            episodic_reward += reward

            train(state_batch, action_batch, reward_batch, next_state_batch, target_actor, actor, target_critic, critic, config["tau"], config["gamma"])

            if done:
                break
            prev_state = state
        ep_reward_list.append(episodic_reward)
        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
import yaml
import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from model.actor import Actor
from model.buffer import Buffer
from model.critic import Critic
from model.second_order_critic import SecondOrderCritic
from random_processes.ornstein_uhlenbeck import OUActionNoise


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def train_mean(state_batch, action_batch, reward_batch, next_state_batch,
               tau, gamma):
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


def train_mean_var(state_batch, action_batch, reward_batch, next_state_batch,
                   tau, gamma, penalty_const):
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
        target_actions = target_actor(next_state_batch, training=True)
        y_square = tf.math.square(reward_batch)\
                   + 2*gamma*target_critic([next_state_batch, target_actions], training=True)\
                   + gamma*gamma*target_second_order([next_state_batch, target_actions], training=True)
        second_order_value = second_order([state_batch, action_batch], training=True)
        second_order_loss = tf.math.reduce_mean(tf.math.square(y_square - second_order_value))
    second_order_grad = tape.gradient(second_order_loss, second_order.trainable_variables)
    second_order_optimizer.apply_gradients(
        zip(second_order_grad, second_order.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor(state_batch, training=True)
        critic_value = critic([state_batch, actions], training=True)
        second_order_value = second_order([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        var_critic_value = second_order_value - tf.math.square(critic_value)
        std_critic_value = tf.math.sqrt(tf.clip_by_value(var_critic_value, 0, var_critic_value.dtype.max))
        actor_loss = -tf.math.reduce_mean(critic_value - penalty_const*std_critic_value)

    actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor.trainable_variables)
    )

    update_target(target_actor.variables, actor.variables, tau)
    update_target(target_critic.variables, critic.variables, tau)
    update_target(target_second_order.variables, second_order.variables, tau)
    return critic_loss, second_order_loss, actor_loss



if __name__ == '__main__':
    # 2 copies of actors and critics
    actor = Actor()
    critic = Critic()
    second_order = SecondOrderCritic()
    target_actor = Actor()
    target_critic = Critic()
    target_second_order = SecondOrderCritic()
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())
    target_second_order.set_weights(second_order.get_weights())

    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

    # problem = "MountainCarContinuous-v0"
    problem = "Pendulum-v1"
    env = gym.make(problem)

    penalty_const = 1.6
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    std_reward_list = []
    tgt_func_list = []

    pred_vals = []
    pred_stds = []
    pred_tgts = []
    pred_second_orders = []

    critic_loss_vals = []
    second_order_loss_vals = []
    actor_loss_vals = []

    with open(f"{problem}.yaml", "r") as f:
        config = yaml.safe_load(f)

    actor.params_from_config(config)
    target_actor.params_from_config(config)

    actor_optimizer = tf.keras.optimizers.Adam(config["actor_lr"])
    critic_optimizer = tf.keras.optimizers.Adam(config["critic_lr"])
    second_order_optimizer = tf.keras.optimizers.SGD()
    buffer = Buffer(config)

    # Takes about 4 min to train
    episodes = config["total_episodes"]
    method = "mean_var"
    # method = "mean_only"
    init_state_batch = tf.transpose(tf.expand_dims(env.reset(),1))
    for ep in range(episodes):

        prev_state = env.reset()
        episodic_reward = 0
        i = 0
        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = actor.get_action(tf_prev_state, ou_noise)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step([action])
            buffer.record((prev_state, action, reward, state))
            state_batch, action_batch, reward_batch, next_state_batch = buffer.replay_sample()
            episodic_reward += reward
            if method == "mean_only":
                train_mean(state_batch, action_batch, reward_batch, next_state_batch,
                               config["tau"], config["gamma"])
            else:
                critic_loss, second_order_loss, actor_loss = train_mean_var(state_batch, action_batch, reward_batch, next_state_batch,config["tau"], config["gamma"], penalty_const)
                critic_loss_vals.append(critic_loss)
                second_order_loss_vals.append(second_order_loss)
                actor_loss_vals.append(actor_loss)
                prev_state = state
                curr_init_action_batch = tf.expand_dims(tf.expand_dims(target_actor.get_action(init_state_batch), -1),
                                                        1)
                pred_val = tf.get_static_value(target_critic([init_state_batch, curr_init_action_batch])[0, 0])
                pred_second_order = tf.get_static_value(
                    target_second_order([init_state_batch, curr_init_action_batch])[0, 0])
                pred_std = np.sqrt(max(pred_second_order - pred_val * pred_val, 0))
                pred_vals.append(pred_val)
                pred_second_orders.append(pred_second_order)
                pred_stds.append(pred_std)
                pred_tgts.append(pred_val - penalty_const * pred_std)
            if done:
                break
        ep_reward_list.append(episodic_reward)
        # Mean of last x episodes
        roll = 20
        avg_reward = np.mean(ep_reward_list[-roll:])
        std_reward = np.std(ep_reward_list[-roll:])
        tgt_func = avg_reward - penalty_const*std_reward
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        print("Episode * {} * Reward Std is ==> {}".format(ep, std_reward))
        print("Episode * {} * Target fun is ==> {}".format(ep, tgt_func))
        avg_reward_list.append(avg_reward)
        std_reward_list.append(std_reward)
        tgt_func_list.append(tgt_func)
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Avg Reward')
    ax1.set_xlabel('Episode')
    ax1.plot(avg_reward_list, label="Avg Reward", color="red")
    ax1.plot(tgt_func_list, label="Std. Adj. Reward", color="green")
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reward Std.')
    ax2.plot(std_reward_list, label="Std Reward", color="blue")
    fig.legend(loc="upper right")
    fig.savefig(f"C:\\Users\\Alice\\Desktop\\学业\\Summer 2022\\DDPG_Learning\\figs\\{problem}\\{method}_penalty_{penalty_const}_{episodes}_rolling{roll}.png")
    plt.close()
    target_actor.save(f"C:\\Users\\Alice\\Desktop\\学业\\Summer 2022\\DDPG_Learning\\trained_models\\{problem}\\{method}_penalty_{penalty_const}_{episodes}_target_actor")
    target_critic.save(f"C:\\Users\\Alice\\Desktop\\学业\\Summer 2022\\DDPG_Learning\\trained_models\\{problem}\\{method}_penalty_{penalty_const}_{episodes}_target_critic")
    if method =="mean_var":
        target_second_order.save(f"C:\\Users\\Alice\\Desktop\\学业\\Summer 2022\\DDPG_Learning\\trained_models\\{problem}\\{method}_penalty_{penalty_const}_{episodes}_target_second_order")
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('Pred Final Reward')
        ax1.set_xlabel('Train Steps')
        ax1.plot(pred_vals[1000:], label="pred valued", color="red")
        ax1.plot(pred_tgts[1000:], label="pred tgt", color="green")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Pred Reward Std.')
        ax2.plot(pred_second_orders[1000:], label="pred second Order", color="blue")
        fig.legend(loc="upper right")
        # fig.savefig(f"C:\\Users\\Alice\\Desktop\\学业\\Summer 2022\\DDPG_Learning\\figs\\mean_var_penalty_{penalty_const}.png")
        fig.savefig(
            f"C:\\Users\\Alice\\Desktop\\学业\\Summer 2022\\DDPG_Learning\\figs\\{problem}\\{method}_penalty_{penalty_const}_{episodes}_pred.png")
        plt.close()

        fig, ax1 = plt.subplots()
        ax1.set_ylabel('Critic Loss')
        ax1.set_xlabel('Train Steps')
        ax1.plot(critic_loss_vals[10:], label="critic_loss_vals", color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Second order loss')
        ax2.plot(second_order_loss_vals[10:], label="Second Order", color="blue")
        fig.legend(loc="upper right")
        fig.savefig(f"C:\\Users\\Alice\\Desktop\\学业\\Summer 2022\\DDPG_Learning\\figs\\{problem}\\{method}_penalty_{penalty_const}_{episodes}_loss_vals.png")
        plt.close()

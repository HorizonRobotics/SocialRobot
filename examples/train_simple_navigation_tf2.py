# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
"""
A demonstration of Q learning for simple_navigation environment
It is the same as train_simple_navigation.py but implemented in tensorflow 2.0

"""

import gym
import os
import random
import social_bot
import logging
import matplotlib.pyplot as plt
import numpy as np
import psutil
import PIL
from social_bot.util.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from collections import deque, namedtuple
import time
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


class Options(object):
    """
    The class for all the settings
    """
    max_steps = int(1e8)
    learning_rate = 5e-4
    history_length = 2
    replay_buffer_size = 500000
    discount_factor = 0.99
    resized_image_size = (84, 84)

    # use nstep reward for updating Q values
    nstep_reward = 10

    # update model every so many steps
    learn_freq = 4

    # starts to update model after so many steps
    learn_start = 8000

    batch_size = 64

    # update Q value target net every so many steps
    target_net_update_freq = 40000

    # exploration linearly decreases from exploration_start to exploration_end
    # in the first exploration_steps steps
    exploration_steps = 500000
    exploration_start = 0.9
    exploration_end = 0.01

    # function for converting action to feature
    # argument _ is for self, because f_action_feature is treated as a class method
    f_action_feature = lambda _, action: (0.5 * (action // 5) - 1, 0.5 * (action % 5) - 1)
    f_action_to_control = lambda _, action: (0.05 * (action // 5) - 0.1, 0.05 * (action % 5) - 0.1)
    action_stand_still = 12
    action_discretize_levels = 5

    # If greater than 0, we calculate the exponential moving average of discounted reward.
    # And use it as baseline for q values.
    ema_reward_alpha = 1. - 1e-5

    # f_action_feature = lambda _, action: (0.4 * (action // 6) - 1, 0.4 * (action % 6) - 1)
    # f_action_to_control = lambda _, action: (0.04 * (action // 6), 0.04 * (action % 6))
    # action_stand_still = 0
    # action_discretize_levels = 6

    # Prioritized Experience Replay: https://arxiv.org/pdf/1511.05952.pdf
    use_prioritized_replay = False
    prioritized_replay_eps = 1e-6
    prioritized_replay_alpha = 0.5
    prioritized_replay_beta0 = 0.3

    # Gamma is for a new term which gives higher priority to experiences near reward.
    # It scales the priority from above by (1+d)**(-gamma), where d is how many steps in
    # the future a non-zero rewad will be encountered. It is gamma linearly decreases
    # from gamma0 to 0 towards the end of the training.
    prioritized_replay_gamma0 = 0.3

    log_freq = 1000
    save_freq = 100000
    model_dir = '/tmp/train_simple_navigation/ema_r_10step'

    show_param_stats_freq = 10000


def main(options):
    """
    The entrance of the program

    Args:
        options (Options): options
    """
    for attr in dir(options):
        if not attr.startswith('__'):
            logging.info(" %s=%s" % (attr, options.__getattribute__(attr)))
    env = gym.make("SocialBot-SimpleNavigation-v0")
    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)
    image_shape = env.observation_space.shape
    agent = QAgent(
        image_shape=(image_shape[2], ) + options.resized_image_size,
        num_actions=options.action_discretize_levels**2,
        options=options)
    episode_rewards = deque(maxlen=options.log_freq)
    steps = deque(maxlen=options.log_freq)
    end_q_values = deque(maxlen=options.log_freq)
    total_steps = 0
    episodes = 0
    t0 = time.time()

    proc = psutil.Process(os.getpid())
    obs = env.reset()
    agent.start_new_episode()
    episode_reward = 0.
    episode_steps = 0
    reward = 0
    period_reward = 0

    logging.info(" mem=%dM" % (proc.memory_info().rss // 1e6))

    while total_steps < options.max_steps:
        obs = PIL.Image.fromarray(obs).resize(options.resized_image_size,
                                              PIL.Image.ANTIALIAS)
        obs = np.transpose(obs, [2, 0, 1])
        action, q = agent.act(obs, reward)
        control = options.f_action_to_control(action)
        new_obs, reward, done, _ = env.step(control)
        agent.learn(obs, action, reward, done)
        obs = new_obs
        episode_reward += reward
        period_reward += reward
        episode_steps += 1
        total_steps += 1
        if done:
            episodes += 1
            episode_rewards.append(episode_reward)
            steps.append(episode_steps)
            end_q_values.append(q)
            reward = 0
            episode_reward = 0.
            episode_steps = 0
            obs = env.reset()
            agent.start_new_episode()

        if total_steps % options.log_freq == 0 and len(steps) > 0 and len(
                episode_rewards) > 0:
            logging.info(
                " episodes=%s" % episodes + " total_steps=%s" % total_steps +
                " fps=%.2f" % (options.log_freq / (time.time() - t0)) +
                " mem=%dM" % (proc.memory_info().rss // 1e6) +
                " r_per_step=%.3g" % (period_reward / options.log_freq) +
                " r_per_episode=%.3g" %
                (sum(episode_rewards) / len(episode_rewards)) +
                " avg_steps=%.3g" % (sum(steps) / len(steps)) +
                " avg_end_q=%.3g" % (sum(end_q_values) / len(steps)) +
                " max_end_q=%.3g" % max(end_q_values) +
                " min_end_q=%.3g" % min(end_q_values) + agent.get_stats())
            period_reward = 0
            agent.reset_stats()
            steps.clear()
            episode_rewards.clear()
            end_q_values.clear()
            t0 = time.time()

        if episodes % options.save_freq == 0:
            agent.save_model(options.model_dir + '/agent.model')


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "reward_dist"])


def select(mat, indices):
    sel = tf.concat([
        tf.reshape(tf.range(mat.shape[0], dtype=tf.int64), (-1, 1)),
        tf.reshape(tf.cast(indices, dtype=tf.int64), (-1, 1))
    ],
                    axis=1)
    return tf.gather_nd(mat, sel)


class QAgent(object):
    """
    A simple Q learning agent for discrete action space
    """

    def __init__(self, image_shape, num_actions, options):
        num_image_channels = image_shape[0]
        num_input_channels = num_image_channels * (options.history_length + 1)
        if options.f_action_feature is not None:
            num_input_channels += len(
                (options.f_action_feature)(0)) * options.history_length
        self._num_actions = num_actions
        self._options = options
        input_shape = image_shape[1:] + (num_input_channels, )

        self._acting_net = Network("acting_net", input_shape, num_actions)
        self._target_net = Network("target_net", input_shape, num_actions)
        self._optimizer = tf.keras.optimizers.Adam(lr=options.learning_rate)
        self._huber_loss = tf.keras.losses.Huber()

        self._episode_steps = 0
        self._total_steps = 0
        C = PrioritizedReplayBuffer if options.use_prioritized_replay else ReplayBuffer
        self._replay_buffer = C(
            options.replay_buffer_size,
            options.history_length,
            future_length=options.nstep_reward)
        self._history = deque(maxlen=options.history_length)
        self.reset_stats()
        self._ema_r = 0.
        self._ema_c = 0.

    def calc_ema_reward(self):
        r = self._ema_r
        f = 1.
        # factor for correcting uncounted future reward
        f -= self._options.discount_factor * self._ema_c
        # factor for correcting limitted steps
        f -= self._options.ema_reward_alpha**self._total_steps
        return r / f

    def act(self, obs, reward):
        """
        Calcuate the action for the current step
        Args:
            obs (np.array): observation for the current step
            reward (float): reward received for the previous step
        Returns:
            int: action id
        """
        eps = self.get_exploration_rate()
        if len(self._history) > 0:
            self._history[-1] = self._history[-1]._replace(reward=reward)
        if self._episode_steps < self._options.history_length:
            action = self._options.action_stand_still
            q = 0
        else:
            input = self._make_input(obs, self._history)
            q_values = self._acting_net.calc_q_values(input)
            q_values = q_values.numpy().reshape(-1)
            if random.random() < eps:
                action = random.randint(0, self._num_actions - 1)
            else:
                action = np.argmax(q_values)
            q = q_values[action]
            if self._options.ema_reward_alpha > 0:
                q += self.calc_ema_reward()
            self._sum_act_q += q
            self._num_act_q += 1

        self._total_steps += 1
        self._episode_steps += 1
        self._history.append(
            Experience(obs, action, reward=0, done=False, reward_dist=0))

        return action, q

    def get_exploration_rate(self):
        p = min(1., float(self._total_steps) / self._options.exploration_steps)
        eps = (1 - p) * self._options.exploration_start \
              + p * self._options.exploration_end
        return eps

    def start_new_episode(self):
        self._episode_steps = 0
        self._history.clear()
        self._ema_c = 0.

    def save_model(self, path):
        self._acting_net.save_weights(path)

    def _get_prioritized_replay_beta(self):
        p = min(1., float(self._total_steps) / self._options.max_steps)
        return (1 - p) * self._options.prioritized_replay_beta0 + p

    def _get_prioritized_replay_gamma(self):
        p = min(1., float(self._total_steps) / self._options.max_steps)
        return (1 - p) * self._options.prioritized_replay_gamma0

    def _update_reward_dist(self):
        i = len(self._replay_buffer) - 2
        d = 1
        indices = []
        priorities = []
        initial_priority = self._replay_buffer.initial_priority
        gamma = self._get_prioritized_replay_gamma()
        while i >= 0:
            e = self._replay_buffer[i]
            if e.reward != 0:
                break
            self._replay_buffer[i] = e._replace(reward_dist=d)
            indices.append(i)
            priorities.append(initial_priority * (1 + d)**(-gamma))
            d += 1
            i -= 1
        self._replay_buffer.update_priority(indices, priorities)

    @tf.function
    def _tf_learn(self, inputs, actions, rewards, next_inputs, dones,
                  is_weights, ema_reward):
        # Double Q Learning: https://arxiv.org/pdf/1509.06461.pdf
        qs_next = self._acting_net.calc_q_values(next_inputs)
        qs_target = self._target_net.calc_q_values(next_inputs)
        a = tf.argmax(qs_next, axis=1)
        q_target = select(qs_target, a)
        q_target = tf.reshape(q_target, (-1, 1)) + ema_reward
        q_target = rewards + (self._options.discount_factor**self._options.
                              nstep_reward) * q_target * (1 - dones)

        with tf.GradientTape() as tape:
            qs = self._acting_net.calc_q_values(inputs)
            q = tf.reshape(select(qs, actions), (-1, 1)) + ema_reward
            td_error = q - q_target
            loss = self._huber_loss(q, q_target, sample_weight=is_weights)

        grads = tape.gradient(loss, self._acting_net.trainable_variables)
        self._optimizer.apply_gradients(
            zip(grads, self._acting_net.trainable_variables))

        return td_error, q, q_target, loss

    def learn(self, obs, action, reward, done):
        """
        Perform one stap of learning

        Args:
            obs (np.array): The observation
            action (int): Action taken at this step
            reward (float): Reward received for this step
            done (bool): Whether reached the end of an episode
        """
        self._ema_c = self._options.ema_reward_alpha * (
            self._options.discount_factor * self._ema_c - 1) + 1
        self._ema_r = self._options.ema_reward_alpha * self._ema_r + self._ema_c * reward
        e = Experience(obs, action, reward, done, reward_dist=0)
        self._replay_buffer.add_experience(e)
        if reward != 0:
            self._update_reward_dist()
        options = self._options
        if self._total_steps <= options.learn_start:
            return
        if self._total_steps % options.learn_freq != 0:
            return

        data = self._get_samples(options.batch_size)

        with tf.device('/device:GPU:0'):
            inputs, actions, rewards, next_inputs, dones, reward_dist, is_weights = map(
                tf.convert_to_tensor, data[:-1])
        indices = data[-1]

        ema_reward = 0.0
        if options.ema_reward_alpha > 0:
            ema_reward = self.calc_ema_reward()
        ema_reward = tf.constant(ema_reward)

        is_weights = is_weights**self._get_prioritized_replay_beta()
        batch_size = options.batch_size

        td_error, q, q_target, loss = self._tf_learn(inputs, actions, rewards,
                                                     next_inputs, dones,
                                                     is_weights, ema_reward)

        # minimize the loss
        priorities = abs(td_error.numpy()).reshape(-1)
        priorities = (priorities + options.prioritized_replay_eps
                      )**options.prioritized_replay_alpha
        gamma = self._get_prioritized_replay_gamma()
        reward_dist = np.reshape(reward_dist, -1)
        priorities = priorities * (1 + reward_dist)**(-gamma)
        self._replay_buffer.update_priority(indices, priorities)

        self._sum_is_weights += np.sum(is_weights)
        self._sum_loss += loss
        self._sum_q += np.mean(q)
        self._sum_q_weighted += np.sum(q * is_weights)
        self._sum_q_target += np.mean(q_target)
        self._sum_q_target_weighted += np.sum(q_target * is_weights)
        self._sum_r += np.mean(rewards)
        self._sum_r_weighted += np.sum(rewards * is_weights)
        self._batches += 1

        if (options.show_param_stats_freq > 0
                and self._total_steps % options.show_param_stats_freq == 0):
            self._acting_net.show_parameter_stats()

        # update target network
        if self._total_steps % options.target_net_update_freq == 0:
            self._target_net.set_weights(self._acting_net.get_weights())

    def get_stats(self):
        """
        Get the internal statistics of this agnet

        Returns
            A string showing all the statistics
        """
        stats = ""
        stats += " exp_rate=%.3g" % self.get_exploration_rate()
        if self._options.ema_reward_alpha > 0:
            stats += " ema_r=%.3g" % (self.calc_ema_reward())
        stats += " avg_act_q=%.3g" % (self._sum_act_q / self._num_act_q)
        if self._batches > 0:
            stats += " avg_loss=%.3g" % (self._sum_loss / self._batches)
            stats += " avg_r=%.3g" % (self._sum_r / self._batches)
            stats += " avg_q=%.3g" % (self._sum_q / self._batches)
            stats += " avg_qt=%.3g" % (self._sum_q_target / self._batches)
        if self._batches > 0 and self._options.use_prioritized_replay:
            stats += " WEIGHTED"
            stats += " avg_loss=%.3g" % (self._sum_loss * self._options.
                                         batch_size / self._sum_is_weights)
            stats += " avg_r=%.3g" % (
                self._sum_r_weighted / self._sum_is_weights)
            stats += " avg_q=%.3g" % (
                self._sum_q_weighted / self._sum_is_weights)
            stats += " avg_qt=%.3g" % (
                self._sum_q_target_weighted / self._sum_is_weights)
        return stats

    def reset_stats(self):
        self._sum_act_q = 0.
        self._num_act_q = 0.
        self._sum_loss = 0.
        self._sum_r = 0.
        self._sum_r_weighted = 0.
        self._sum_q = 0.
        self._sum_q_weighted = 0.
        self._sum_q_target = 0.
        self._sum_q_target_weighted = 0.
        self._sum_is_weights = 0
        self._batches = 0

    def _make_input(self, obs, history):
        def make_action_feature(action):
            af = (self._options.f_action_feature)(action)
            af = np.array(af, dtype=np.float32).reshape(-1, 1, 1)
            return np.broadcast_to(af, (af.shape[0], ) + obs.shape[1:])

        scale = 2. / 255
        features = []
        for e in history:
            features.append(e.state.astype(np.float32) * scale - 1)
            if self._options.f_action_feature:
                features.append(make_action_feature(e.action))
        features.append(obs.astype(np.float32) * scale - 1)
        input = np.vstack(features)
        input = input.transpose([1, 2, 0])
        input = input.reshape((1, ) + input.shape)
        return input

    def _get_samples(self, batch_size):
        """Randomly sample a batch of experiences from memory."""

        def _make_sample(*exps):
            # inputs, actions, rewards, next_inputs, dones
            h = self._options.history_length
            reward = 0
            done = False
            for s in reversed(range(h, h + self._options.nstep_reward)):
                reward = (
                    1 - exps[s].done
                ) * self._options.discount_factor * reward + exps[s].reward
                done = done or exps[s].done
            return (self._make_input(exps[h].state, exps[:h]), exps[h].action,
                    np.float32(reward),
                    self._make_input(exps[-1].state, exps[-(h + 1):-1]),
                    np.float32(done), np.float32(exps[h].reward_dist))

        features, indices, is_weights = self._replay_buffer.get_sample_features(
            self._options.batch_size, _make_sample)
        return features + [is_weights, indices]


class Network(tf.keras.Model):
    """
    The neural network module for calculating the Q values.
    """

    def __init__(self, name, input_shape, num_actions):
        super(Network, self).__init__()

        num_filters = (16, 32)
        fc_size = (64, 64)

        self.latent_nn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                num_filters[0],
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                input_shape=input_shape,
                data_format='channels_last',
            ),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2), data_format='channels_last'),
            tf.keras.layers.Conv2D(
                num_filters[1],
                kernel_size=(3, 3),
                padding='same',
                strides=1,
                data_format='channels_last'),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2), data_format='channels_last'),
            tf.keras.layers.Flatten(data_format='channels_last')
        ],
                                             name=name + "/latent_nn")

        self.latent_nn.summary()

        self.q_nn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                fc_size[0], input_shape=self.latent_nn.output_shape[1:]),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(fc_size[1]),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(num_actions)
        ],
                                        name=name + "/q_nn")
        self.q_nn.summary()

        self.v_nn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                fc_size[0], input_shape=self.latent_nn.output_shape[1:]),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(fc_size[1]),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(1)
        ],
                                        name=name + "/v_nn")
        self.v_nn.summary()

    def calc_q_values(self, state):
        latent = self.latent_nn(state)
        q_values = self.q_nn(latent)

        # Dueling Network: https://arxiv.org/pdf/1511.06581.pdf
        value = self.v_nn(latent)
        mean_q = tf.reduce_mean(q_values, axis=-1, keepdims=True)
        adjust = value - mean_q
        q_values = q_values + adjust
        return q_values

    def call(self, state):
        return self.calc_q_values()

    def show_parameter_stats(self):
        pass


if __name__ == "__main__":
    options = Options()
    os.makedirs(options.model_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    # tensorflow already created a logger, thus previous line won't create new logger
    # hence have no effect.
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(
        logging.FileHandler(filename=options.model_dir + '/train.log'))
    main(options)

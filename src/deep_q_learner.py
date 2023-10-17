import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

class DeepQLearner:
    def __init__(self, env, n_iterations, max_steps, learning_rate, discount_factor, epsilon=0.1, render=False,
                 render_text_file=False, simple_network=True, batch_size=64):

        self.env = env
        self.render = render
        self.render_text_file = render_text_file

        # self.optimizer = Adam()
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        # self.optimizer = RMSprop()
        self.loss_function = keras.losses.MeanSquaredError()
        self.batch_size = batch_size
        self.update_frequency = 1
        # self.update_target_network = 100
        # self.update_target_network = int(n_iterations * 0.1)
        self.update_target_network = 1000
        self.initial_exploratory_steps = int(n_iterations * 0.01)
        # self.initial_exploratory_steps = 100
        self.buffer_size = 1000000
        self.deep_agent = self.__get_deep_agent__(simple_network)
        self.reward_predictor = self.__get_deep_agent__(simple_network)

        self.n_iter = n_iterations
        self.max_steps_per_iteration = max_steps
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.eps_init = 1.0
        self.eps = 1.0
        self.eps_min = epsilon

        # experience replay buffers
        self.action_buffer = []
        self.state_buffer = []
        self.next_state_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.episode_reward_history = []

        # train
        self.rewards = []
        self.train_episode_rewards = []
        self.steps_taken = []
        self.train_treasure_found = 0

        # test
        self.test_rewards = []
        self.test_episode_rewards = []
        self.test_steps_taken = []
        self.test_treasure_found = 0

    def __get_deep_agent__(self, simple=True):
        # input = Input(shape=(1,), batch_size=self.batch_size)
        input = Input(shape=(self.env.size,), batch_size=self.batch_size)

        if simple:
            x = Dense(30, activation='relu')(input)
        else:
            x = Dense(32, activation='relu')(input)
            x = Dense(32, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            # x = Dense(32, activation='relu')(x)

        out = Dense(self.env.action_space.n, activation='linear', name='out')(x)

        return keras.Model(input, out)

    def train(self, render_path='.'):

        total_steps = 0
        running_reward = 0

        # for it in range(self.n_iter):
        while True:

            # if it % 1000 == 0:
            #     print(f'Training iteration {it + 1}/{self.n_iter}')

            timesteps = 0
            self.env.episode_count += 1

            state = self.env.reset()

            decay = max((self.n_iter - self.env.episode_count) / self.n_iter, 0)
            self.eps = (self.eps_init - self.eps_min) * decay + self.eps_min

            # print(f'Epsilon: {self.eps}')

            if self.env.agent_pos == self.env.poison_pos:
                # you lost
                self.steps_taken.append(timesteps)
                self.rewards.append(-10)
                self.train_episode_rewards.append(-10)

                if self.render:
                    self.env.render()
                if self.render_text_file:
                    self.env.render_text_file(render_path)

                continue
            if self.env.agent_pos == self.env.treasure_pos:
                # you won
                self.steps_taken.append(timesteps)
                self.rewards.append(10)
                self.train_episode_rewards.append(10)
                self.train_treasure_found += 1

                if self.render:
                    self.env.render()
                if self.render_text_file:
                    self.env.render_text_file(render_path)

                continue

            episode_reward = 0

            for t in range(self.max_steps_per_iteration):

                timesteps += 1
                total_steps += 1

                if self.render:
                    self.env.render()
                if self.render_text_file:
                    self.env.render_text_file(render_path)

                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)

                if self.eps > np.random.rand(1)[0]:  # or total_steps < self.initial_exploratory_steps:
                    action = int(self.env.action_space.sample())
                else:
                    action_probs = self.deep_agent(state_tensor, training=False)
                    action = int(tf.argmax(action_probs[0]).numpy())

                next_state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)
                episode_reward += reward

                # store experience
                self.action_buffer.append(action)
                self.state_buffer.append(state)
                self.next_state_buffer.append(next_state)
                self.reward_buffer.append(reward)
                self.done_buffer.append(done)
                state = next_state

                if len(self.action_buffer) > self.batch_size:  # and total_steps % self.update_frequency == 0:
                    mini_batch_idxs = np.random.choice(len(self.action_buffer), size=self.batch_size)

                    state_batch = np.array([self.state_buffer[i] for i in mini_batch_idxs])
                    next_state_batch = np.array([self.next_state_buffer[i] for i in mini_batch_idxs])
                    reward_batch = [self.reward_buffer[i] for i in mini_batch_idxs]
                    action_batch = [self.action_buffer[i] for i in mini_batch_idxs]
                    done_batch = tf.convert_to_tensor(
                        [float(self.done_buffer[i]) for i in mini_batch_idxs]
                    )

                    # next_state_batch_tensor = tf.convert_to_tensor(next_state_batch)
                    # future_reward = self.reward_predictor.predict(next_state_batch_tensor)

                    # future_reward = self.reward_predictor.predict(next_state_batch)
                    future_reward = self.deep_agent.predict(next_state_batch)

                    updated_q_values = reward_batch + self.gamma * tf.reduce_max(
                        future_reward, axis=1
                    )
                    updated_q_values = updated_q_values * (1 - done_batch) - done_batch

                    num_actions = 4
                    mask = tf.one_hot(action_batch, num_actions)

                    with tf.GradientTape() as tape:
                        # state_batch_tensor = tf.convert_to_tensor(state_batch)
                        # q_values = self.deep_agent(state_batch_tensor)
                        q_values = self.deep_agent(state_batch)
                        q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
                        loss = self.loss_function(updated_q_values, q_action)

                    grads = tape.gradient(loss, self.deep_agent.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.deep_agent.trainable_variables))

                if total_steps % self.update_target_network == 0:
                    self.reward_predictor.set_weights(self.deep_agent.get_weights())
                    template = "running reward: {:.2f} at episode {}, step count {}, epsilon {}"
                    print(template.format(running_reward, self.env.episode_count, total_steps, self.eps))

                if len(self.action_buffer) > self.buffer_size:
                    del self.action_buffer[:1]
                    del self.state_buffer[:1]
                    del self.next_state_buffer[:1]
                    del self.reward_buffer[:1]
                    del self.done_buffer[:1]

                if done:
                    if self.env.agent_pos == self.env.treasure_pos:
                        self.train_treasure_found += 1

                    break

            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            running_reward = np.mean(self.episode_reward_history)

            self.steps_taken.append(timesteps)
            self.train_episode_rewards.append(episode_reward)

            if self.render:
                self.env.render()
            if self.render_text_file:
                self.env.render_text_file(render_path)

            if running_reward > 5.0:
                print("Solved at episode {}!".format(self.env.episode_count))
                break

        return self.rewards, self.train_episode_rewards, self.steps_taken, self.train_treasure_found

    def test(self, render_path='.', n_runs=1):
        self.env.episode_count = 0

        for it in range(n_runs):
            # print(f'Test episode {it + 1} of {n_runs}')
            self.env.episode_count += 1
            timesteps = 0
            final_reward = 0
            state = self.env.reset()

            if self.env.agent_pos == self.env.poison_pos:
                # you lost
                final_reward = -10
                self.test_steps_taken.append(timesteps)
                self.test_rewards.append(-10)
                self.test_episode_rewards.append(-10)

                if self.render:
                    self.env.render()
                if self.render_text_file:
                    self.env.render_text_file(render_path)

                continue
            if self.env.agent_pos == self.env.treasure_pos:
                # you won
                final_reward = 10
                self.test_steps_taken.append(timesteps)
                self.test_rewards.append(10)
                self.test_episode_rewards.append(10)

                if self.render:
                    self.env.render()
                if self.render_text_file:
                    self.env.render_text_file(render_path)

                continue

            episode_reward = 0

            done = False
            for _ in range(100):
                timesteps += 1

                if self.render:
                    self.env.render()
                if self.render_text_file:
                    self.env.render_text_file(render_path)

                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.deep_agent(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()

                next_state, reward, done, info = self.env.step(action)
                self.test_rewards.append(reward)
                episode_reward += reward

                final_reward += reward
                state = next_state

                if done:
                    if self.env.agent_pos == self.env.treasure_pos:
                        self.test_treasure_found += 1

                    break

            self.test_steps_taken.append(timesteps)
            self.test_episode_rewards.append(episode_reward)

            if self.render:
                self.env.render()
            if self.render_text_file:
                self.env.render_text_file(render_path)

        return self.test_rewards, self.test_episode_rewards, self.test_steps_taken, self.test_treasure_found

import numpy as np


# q-learning: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
class QLearner:
    def __init__(self, env, n_iterations, max_steps, learning_rate, discount_factor, epsilon=0.1, render_mode="text", render_text_file=False):
        self.env = env
        self.q_table = np.zeros([env.size, env.action_space.n])
        self.n_iter = n_iterations
        self.max_steps_per_iteration = max_steps
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.eps = epsilon

        # render options
        self.render_mode = render_mode
        self.render_text_file = render_text_file

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

    def train(self, experiment_name='train_experiment'):
        self.env.episode_count = 0

        for it in range(self.n_iter):
            self.env.episode_count += 1
            timesteps = 0
            state = self.env.reset()

            if state[0] == state[2]:
                # you lost
                self.steps_taken.append(timesteps)
                self.rewards.append(-10)
                self.train_episode_rewards.append(-10)

                self.env.render(self.render_mode)
                if self.render_text_file:
                    self.env.render_text_file(experiment_name)

                continue

            if state[0] == state[3]:
                # you won
                self.steps_taken.append(timesteps)
                self.rewards.append(10)
                self.train_episode_rewards.append(10)
                self.train_treasure_found += 1

                self.env.render(self.render_mode)
                if self.render_text_file:
                    self.env.render_text_file(experiment_name)

                continue

            episode_reward = 0

            for t in range(self.max_steps_per_iteration):

                timesteps += 1
                self.env.render(self.render_mode)
                if self.render_text_file:
                    self.env.render_text_file(experiment_name)

                if np.random.uniform() < self.eps:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state[0]])

                next_state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)
                episode_reward += reward

                old_value = self.q_table[state[0], action]
                next_max = np.max(self.q_table[next_state[0]])

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state[0], action] = new_value

                state = next_state

                if done:
                    if state[0] == state[3]:
                        self.train_treasure_found += 1

                    break

            self.steps_taken.append(timesteps)
            self.train_episode_rewards.append(episode_reward)

            self.env.render(self.render_mode)
            if self.render_text_file:
                self.env.render_text_file(experiment_name)

        return self.rewards, self.train_episode_rewards, self.steps_taken, self.train_treasure_found

    def test(self, experiment_name='test_experiment', n_runs=1):
        self.env.episode_count = 0

        for it in range(n_runs):
            self.env.episode_count += 1
            timesteps = 0
            final_reward = 0
            state = self.env.reset()

            episode_reward = 0

            if state[0] == state[2]:
                # you lost
                final_reward = -10
                # print('you started in the losing cell')
                self.test_steps_taken.append(timesteps)
                self.test_rewards.append(-10)
                self.test_episode_rewards.append(-10)

                self.env.render(self.render_mode)
                if self.render_text_file:
                    self.env.render_text_file(experiment_name)

                continue

            if state[0] == state[3]:
                # you won
                final_reward = 10
                # print('you started in the winning cell')
                self.test_steps_taken.append(timesteps)
                self.test_rewards.append(10)
                self.test_episode_rewards.append(10)
                self.test_treasure_found += 1

                self.env.render(self.render_mode)
                if self.render_text_file:
                    self.env.render_text_file(experiment_name)

                continue

            done = False
            for _ in range(100):
                timesteps += 1
                self.env.render(self.render_mode)
                if self.render_text_file:
                    self.env.render_text_file(experiment_name)

                action = np.argmax(self.q_table[state[0]])

                next_state, reward, done, info = self.env.step(action)
                self.test_rewards.append(reward)
                episode_reward += reward

                final_reward += reward
                state = next_state

                if done:
                    if state[0] == state[3]:
                        self.test_treasure_found += 1
                    break

            self.test_episode_rewards.append(episode_reward)
            self.test_steps_taken.append(timesteps)

            self.env.render(self.render_mode)
            if self.render_text_file:
                self.env.render_text_file(experiment_name)

        return self.test_rewards, self.test_episode_rewards, self.test_steps_taken, self.test_treasure_found
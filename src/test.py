import numpy as np
from environment import TreasureRoom
from q_learner import QLearner
from utils import GIF_generator

def main():
    N = 5
    n_iterations = 100
    env = TreasureRoom(N, verbose=0)
    q_learner = QLearner(env=env,
                         n_iterations=n_iterations,
                         max_steps=100,
                         learning_rate=0.1,
                         discount_factor=0.99,
                         epsilon=0.1,
                         render_mode="gui",
                         render_text_file=False)

    # train ###
    experiment_name = "exp1_train"
    train_reward, train_episode_reward, train_steps, train_treasures = q_learner.train(experiment_name=experiment_name)
    print(f'Training mean reward: {np.mean(train_reward)}')
    print(f'Training episode mean reward: {np.mean(train_episode_reward)}')
    print(f'Training mean timesteps: {np.mean(train_steps)}')
    print(f'Train treasure rate: {train_treasures / n_iterations}') # denominator is n_iterations

    train_gif = GIF_generator(experiment_name=experiment_name)
    train_gif.get_gif_from_images()

    # test ###
    n_runs = 10
    experiment_name = "exp1_test"
    test_reward, test_episode_reward, test_steps, test_treasures = q_learner.test(experiment_name=experiment_name, n_runs=n_runs)
    print(f'Testing mean reward: {np.mean(test_reward)}')
    print(f'Testing episode mean reward: {np.mean(test_reward)}')
    print(f'Testing mean timesteps: {np.mean(test_steps)}')
    print(f'Test treasure rate: {test_treasures / n_runs}')

    test_gif = GIF_generator(experiment_name=experiment_name)
    test_gif.get_gif_from_images()


if __name__ == "__main__":
    main()
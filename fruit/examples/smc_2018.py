from fruit.agents.factory import A3CAgent
from fruit.envs.ale import ALEEnvironment
from fruit.networks.policy import PolicyNetwork
from fruit.networks.policy import DQPolicyNetwork
from fruit.networks.config.atari import AtariA3CConfig
from fruit.networks.config.atari import AtariDQA3CConfig
from fruit.utils.processor import AtariBlackenProcessor
from stat import ST_CTIME
import os
import matplotlib.pyplot as plt
import numpy as np


def train_breakout_with_a3c_remove_immutable_objects():

    # Create an ALE for game Breakout, blacken top half of the state
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT,
                                 loss_of_life_negative_reward=True,
                                 state_processor=AtariBlackenProcessor())

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, num_of_checkpoints=20)

    # Create A3C agent
    agent = A3CAgent(network, environment, num_of_epochs=70, steps_per_epoch=1e6, save_frequency=5e6,
                     log_dir='./train/smc/Breakout/a3c_gpu_8_threads_breakout_time_based_remove_objects_loss_life_negative_reward',
                     num_of_threads=8)

    # Train it
    agent.train()


def train_breakout_with_a3c_loss_of_life_reward():

    # Create an ALE for game Breakout. negative reward for each lost life
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT, loss_of_life_negative_reward=True)

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, num_of_checkpoints=20)

    # Create A3C agent
    agent = A3CAgent(network, environment, num_of_epochs=70, steps_per_epoch=1e6, save_frequency=5e6,
                     log_dir='./train/smc/Breakout/a3c_gpu_8_threads_breakout_time_based_normal_loss_life_negative_reward',
                     num_of_threads=8)

    # Train it
    agent.train()


def train_breakout_with_a3c_normal():
    # Create an ALE for game Breakout. negative reward for each lost life
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, num_of_checkpoints=20)

    # Create A3C agent
    agent = A3CAgent(network, environment, num_of_epochs=70, steps_per_epoch=1e6, save_frequency=5e6,
                     log_dir='./train/smc/Breakout/a3c_gpu_8_threads_breakout_time_based_normal',
                     num_of_threads=8)

    # Train it
    agent.train()


def composite_agents(path_1, path_2, alpha, epsilon):

    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)

    network_config = AtariDQA3CConfig(environment)

    network = DQPolicyNetwork(network_config, load_model_path=path_1,
                              load_model_path_2=path_2,
                              alpha=alpha,
                              epsilon=epsilon)

    agent = A3CAgent(network, environment,
                     num_of_threads=8,
                     report_frequency=1,
                     num_of_epochs=12,
                     steps_per_epoch=10000,
                     log_dir="./thi_test/Lifetime/Breakout/a3c")

    return agent.evaluate()


def perform_evaluation(alpha, epsilon):
    print("alpha", alpha, "epsilon", epsilon)
    log_dir = "./thi_test/smc/"
    model_path_a3c_normal = "./train/smc/Breakout/a3c_gpu_8_threads_breakout_time_based_normal_03-24-2018-15-05"
    model_names_a3c_normal = []
    for model in os.listdir(model_path_a3c_normal):
        if model.endswith(".meta"):
            full_path = os.path.join(model_path_a3c_normal, model)
            model_names_a3c_normal.append((os.stat(full_path)[ST_CTIME], full_path))
    model_names_a3c_normal.sort()

    model_path_a3c_life_greedy = "./train/smc/Breakout/a3c_gpu_8_threads_breakout_time_based_remove_objects_loss_life_negative_reward_03-22-2018-19-06"
    model_names_life_greedy = []
    for model in os.listdir(model_path_a3c_life_greedy):
        if model.endswith(".meta"):
            full_path = os.path.join(model_path_a3c_life_greedy, model)
            model_names_life_greedy.append((os.stat(full_path)[ST_CTIME], full_path))
    model_names_life_greedy.sort()

    max_rewards = []
    mean_rewards = []
    min_rewards = []
    steps_test = []
    local_stuck_times = []
    steps_count = []
    count = 0
    index = 0
    for _, model in model_names_a3c_normal:
        reward_list = composite_agents(model[:-5], model_names_life_greedy[index][1][:-5], alpha, epsilon)
        index = index + 1

        max_reward = max([x[0] for x in reward_list])
        mean_reward = np.median([x[0] for x in reward_list])
        min_reward = min([x[0] for x in reward_list])
        step = np.mean([x[2] for x in reward_list])

        max_rewards.append(max_reward)
        mean_rewards.append(mean_reward)
        min_rewards.append(min_reward)
        steps_count.append(step)

        count = count + 1
        steps_test.append(5 * count)

        episode_steps = [x[2] for x in reward_list]
        local_stuck = 0
        for j in episode_steps:
            if j > 10000:
                local_stuck = local_stuck + 1
        local_stuck_times.append(local_stuck)

        print("################## RESULT ###################")
        print("Max Reward:", max_reward)
        print("Median Reward:", mean_reward)
        print("Min Reward:", min_reward)
        print("Local Stuck:", local_stuck)
        print("Training steps: {0}".format(5 * count))
        print("#############################################")

    plt.plot(steps_test, mean_rewards, 'go-', label='Median reward', linewidth=1)
    plt.plot(steps_test, max_rewards, 'rs-', label='Max Reward', linewidth=1)
    plt.plot(steps_test, min_rewards, 'bx-', label='Min Reward', linewidth=1)
    plt.legend(loc='best')
    plt.savefig(log_dir + '/performance_test_stochastic_' + str(alpha) + '_' + str(epsilon) + '.png')
    plt.show()

    return steps_test, steps_count


if __name__ == '__main__':

    # train_breakout_with_a3c_remove_immutable_objects()

    # train_breakout_with_a3c_normal()

    log_dir = "./thi_test/smc/"
    alpha = 0.
    steps_test = None
    steps_list = []
    epsilon = 0.01
    for i in range(0, 9):
        steps_test, total_steps = perform_evaluation(alpha, epsilon)
        alpha = alpha + 0.125
        steps_list.append(total_steps)

        print(steps_list)

    plt.plot(steps_test, steps_list[0], 'go-', linewidth=1)
    plt.plot(steps_test, steps_list[1], 'rs-', linewidth=1)
    plt.plot(steps_test, steps_list[3], 'bx-', linewidth=1)
    plt.plot(steps_test, steps_list[6], 'm+-', linewidth=1)
    plt.plot(steps_test, steps_list[8], 'c^-', linewidth=1)
    plt.savefig(log_dir + '/performance_test_stochastic_number_steps_' + str(epsilon) + '.png')
    plt.show()
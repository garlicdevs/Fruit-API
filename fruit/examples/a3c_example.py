from fruit.agents.factory import A3CAgent
from fruit.agents.factory import A3CLSTMAgent
from fruit.envs.ale import ALEEnvironment
from fruit.networks.policy import PolicyNetwork
from fruit.networks.config.atari import AtariA3CConfig
from fruit.networks.config.atari import AtariA3CLSTMConfig
from stat import ST_CTIME
import os
import matplotlib.pyplot as plt
import numpy as np


def train_ale_environment():
    # Create an ALE for Breakout
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)
    ###################################################################################
    # In the case of using Gym
    # environment = GymEnvironment("Breakout-v0", state_processor=AtariProcessor())
    ###################################################################################

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, num_of_checkpoints=40, using_gpu=True)

    # Create A3C agent
    agent = A3CAgent(network, environment, save_time_based=30, checkpoint_stop=20,
                     log_dir='./train/Breakout/a3c_gpu_8_threads_breakout_time_based_30_20', num_of_threads=8)

    # Train it
    agent.train()


def train_ale_environment_lstm():

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.PONG, loss_of_life_termination=True)
    ###################################################################################
    # In case using Gym
    # environment = GymEnvironment("Breakout-v0", state_processor=AtariProcessor())
    ###################################################################################

    # Create a network configuration for Atari A3C
    network_config = AtariA3CLSTMConfig(environment, initial_learning_rate=0.001)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, num_of_checkpoints=40, using_gpu=True)

    # Create A3C agent
    agent = A3CLSTMAgent(network, environment, save_time_based=30, checkpoint_stop=40,
                     log_dir='./train/Pong/a3c_lstm_gpu_8_threads_pong_time_based_30_20', num_of_threads=8)

    # Train it
    agent.train()


def eval_ale_environment(model_path, render, num_of_epochs, steps_per_epoch, stochastic, initial_epsilon, log_dir):

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.PONG, is_render=render, max_episode_steps=5000)

    # Create a network configuration for Atari A3C
    if not stochastic:
        network_config = AtariA3CConfig(environment, stochastic=False)

        # Create a shared network for A3C agent
        network = PolicyNetwork(network_config, load_model_path=model_path)

        # Create A3C agent
        agent = A3CAgent(network, environment, num_of_threads=8, using_e_greedy=True,
                         initial_epsilon=initial_epsilon, report_frequency=1,
                         num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir)
    else:
        network_config = AtariA3CConfig(environment)

        # Create a shared network for A3C agent
        network = PolicyNetwork(network_config, load_model_path=model_path)

        # Create A3C agent
        agent = A3CAgent(network, environment, num_of_threads=8, report_frequency=1,
                         num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir)

    # Evaluate it
    return agent.evaluate()


def performance_test(stochastic=False, epsilon=0.02):

    log_dir = "./test/a3c_test_1_cpu/"
    model_path = "./train/a3c_cpu_1_threads_breakout_time_based_30_32_02-11-2018-16-17"
    model_names = []
    for model in os.listdir(model_path):
        if model.endswith(".meta"):
            full_path = os.path.join(model_path, model)
            model_names.append((os.stat(full_path)[ST_CTIME], full_path))

    model_names.sort()

    max_rewards = []
    mean_rewards = []
    min_rewards = []
    time_test = []
    local_stuck_times = []
    count = 0
    for _, model in model_names:
        reward_list = eval_ale_environment(model[:-5], render=False, num_of_epochs=1, steps_per_epoch=100000,
                                             stochastic=stochastic, initial_epsilon=epsilon, log_dir=log_dir)

        max_reward = max([x[0] for x in reward_list])
        mean_reward = np.mean([x[0] for x in reward_list])
        min_reward = min([x[0] for x in reward_list])

        max_rewards.append(max_reward)
        mean_rewards.append(mean_reward)
        min_rewards.append(min_reward)
        count = count + 1
        time_test.append(0.5*count)

        episode_steps = [x[2] for x in reward_list]
        total_local_minimum = 0
        for j in episode_steps:
            if j > 5000:
                total_local_minimum = total_local_minimum + 1
        local_stuck_times.append(total_local_minimum)

        print("################## RESULT ###################")
        print("Max Reward:", max_reward)
        print("Mean Reward:", mean_reward)
        print("Min Reward:", min_reward)
        print("Local Stuck:", total_local_minimum)
        print("Training time: {0} (hours)".format(0.5 * count))
        print("#############################################")

    plt.plot(time_test, mean_rewards, 'go-', label='Mean Reward', linewidth=1)
    plt.plot(time_test, max_rewards, 'rs-', label='Max Reward', linewidth=1)
    plt.plot(time_test, min_rewards, 'bx-', label='Min Reward', linewidth=1)
    plt.legend(loc='best')
    if stochastic:
        plt.savefig(log_dir + '/performance_test_stochastic.png')
    else:
        plt.savefig(log_dir + '/performance_test_deterministic_epsilon_'+ str(epsilon) +'.png')
    plt.show()

    plt.plot(time_test, local_stuck_times, 'rs-', label='Local Stuck Times', lineWidth=1)
    plt.legend(loc='best')
    if stochastic:
        plt.savefig(log_dir + '/performance_test_stochastic_local_stuck.png')
    else:
        plt.savefig(log_dir + '/performance_test_deterministic_epsilon_'+ str(epsilon) +'_local_stuck.png')
    plt.show()


def performance_test_2():
    log_dir = "./test/a3c_test/"
    model_path = "./train/a3c_breakout_time_based_30_32_02-02-2018-15-53/model-49861769"
    mean_rewards = []
    local_stuck_times = []
    max_rewards = []
    min_rewards = []

    # Deterministic test
    init_epsilon = 0
    eps = []
    for i in range(20):
        eps.append(init_epsilon)

        reward_list = eval_ale_environment(model_path=model_path, render=False, num_of_epochs=1, steps_per_epoch=100000,
                                           stochastic=False, initial_epsilon=init_epsilon, log_dir=log_dir)

        max_reward = max([x[0] for x in reward_list])
        mean_reward = np.mean([x[0] for x in reward_list])
        min_reward = min([x[0] for x in reward_list])

        max_rewards.append(max_reward)
        mean_rewards.append(mean_reward)
        min_rewards.append(min_reward)

        episode_steps = [x[2] for x in reward_list]
        total_local_minimum = 0
        for j in episode_steps:
            if j > 5000:
                total_local_minimum = total_local_minimum + 1
        local_stuck_times.append(total_local_minimum)

        print("################## RESULT ###################")
        print("Max Reward:", max_reward)
        print("Mean Reward:", mean_reward)
        print("Min Reward:", min_reward)
        print("Epsilon:", init_epsilon)
        print("Local Stuck:", total_local_minimum)
        print("#############################################")

        init_epsilon = init_epsilon + 0.005

    plt.plot(eps, mean_rewards, 'go-', label='Mean Reward', linewidth=1)
    plt.plot(eps, max_rewards, 'rs-', label='Max Reward', linewidth=1)
    plt.plot(eps, min_rewards, 'bx-', label='Min Reward', linewidth=1)
    plt.legend(loc='best')
    plt.savefig(log_dir + '/performance_test_deterministic_epsilon_0_0.treasure.png')
    plt.show()

    plt.plot(eps, local_stuck_times, 'rs-', label='Local Stuck Times', lineWidth=1)
    plt.legend(loc='best')
    plt.savefig(log_dir + '/performance_test_deterministic_epsilon_0_0.1_local_stuck.png')
    plt.show()


def performance_test_3():
    log_dir = "./test/a3c_test/"
    model_path = "./train/a3c_breakout_time_based_30_32_02-02-2018-15-53/model-49861769"
    mean_rewards = []
    local_stuck_times = []
    max_rewards = []
    min_rewards = []

    # Stochastic test
    reward_list = eval_ale_environment(model_path=model_path, render=False, num_of_epochs=1, steps_per_epoch=100000,
                                           stochastic=True, initial_epsilon=0.01, log_dir=log_dir)

    max_reward = max([x[0] for x in reward_list])
    mean_reward = np.mean([x[0] for x in reward_list])
    min_reward = min([x[0] for x in reward_list])

    max_rewards.append(max_reward)
    mean_rewards.append(mean_reward)
    min_rewards.append(min_reward)

    episode_steps = [x[2] for x in reward_list]
    total_local_minimum = 0
    for j in episode_steps:
        if j > 5000:
            total_local_minimum = total_local_minimum + 1
    local_stuck_times.append(total_local_minimum)

    print("################## RESULT ###################")
    print("Max Reward:", max_reward)
    print("Mean Reward:", mean_reward)
    print("Min Reward:", min_reward)
    print("Local Stuck:", total_local_minimum)
    print("#############################################")

    ################## RESULT ###################
    # Max Reward: 855
    # Mean Reward: 446.837837838
    # Min Reward: 365
    # Local Stuck: 4
    #############################################

    # environment = ALEEnvironment(ALEEnvironment.PONG, is_render=True, max_episode_steps=5000)
    # Create a network configuration for Atari A3C
    # network_config = AtariA3CConfig(environment)
    # Create a shared network for A3C agent
    # network = PolicyNetwork(network_config,
    #                        load_model_path="./train/Pong/a3c_gpu_8_threads_pong_time_based_30_20_02-12-2018-23-08/model-10073882")
    # Create A3C agent
    # agent = A3CAgent(network, environment, num_of_threads=1, report_frequency=1,
    #                 num_of_epochs=10, steps_per_epoch=100000, log_dir="./test/Pong/a3c_gpu_8_threads")
    # Evaluate it
    # agent.evaluate()


if __name__ == '__main__':

    #train_ale_environment()

    #performance_test(stochastic=False, epsilon=0.02)

    #performance_test_2()

    #performance_test_3()

    #train_ale_environment_lstm()

    train_ale_environment()
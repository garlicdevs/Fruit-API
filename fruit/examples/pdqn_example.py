from fruit.agents.dqn import DQNAgent
from fruit.envs.ale import ALEEnvironment
from fruit.networks.policy import PolicyNetwork
from fruit.networks.config.atari import PrioritizedAtariDQNConfig
from stat import ST_CTIME
import os
import matplotlib.pyplot as plt
import numpy as np


# DQN with Prioritized Experience Replay
def train_ale_environment():

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)
    ###################################################################################
    # In case using Gym
    # environment = GymEnvironment("Breakout-v0", state_processor=AtariProcessor())
    ###################################################################################

    # Create a network configuration for Atari DQN
    network_config = PrioritizedAtariDQNConfig(environment, initial_beta=0.4,
                                               initial_learning_rate=0.00025,
                                               debug_mode=True)

    # Create a policy network for DQN agent
    network = PolicyNetwork(network_config, max_num_of_checkpoints=100)

    # Create DQN agent
    agent = DQNAgent(network, environment,
                     save_frequency=5e5,
                     steps_per_epoch=1e6,
                     num_of_epochs=50,
                     exp_replay_size=2**19,
                     importance_sampling=True,
                     log_dir="./train/breakout/pdqn_check_point",
                     prioritized_alpha=0.6, prioritized=True)

    # Train it
    agent.train()


def eval_ale_environment(model_path, render, num_of_epochs, steps_per_epoch, initial_epsilon, log_dir):

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.RIVERRAID, is_render=render, max_episode_steps=5000)

    # Create a network configuration for Atari DQN
    network_config = PrioritizedAtariDQNConfig(environment)

    # Create a policy network for DQN agent
    network = PolicyNetwork(network_config, load_model_path=model_path)

    # Create DQN agent
    agent = DQNAgent(network, environment, initial_epsilon=initial_epsilon, report_frequency=1, num_of_threads=8,
                     num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir, prioritized=True)

    # Evaluate it
    return agent.evaluate()


def performance_test(epsilon=0.02):

    log_dir = "./thi_test/pdqn_test/"
    model_path = "./train/pdqn_breakout_time_based_30_40_02-10-2018-00-26"
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
                                           initial_epsilon=epsilon, log_dir=log_dir)

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
    plt.savefig(log_dir + '/performance_test_deterministic_epsilon_'+ str(epsilon) +'.png')
    plt.show()

    plt.plot(time_test, local_stuck_times, 'rs-', label='Local Stuck Times', lineWidth=1)
    plt.legend(loc='best')
    plt.savefig(log_dir + '/performance_test_deterministic_epsilon_'+ str(epsilon) +'_local_stuck.png')
    plt.show()


if __name__ == '__main__':
    train_ale_environment()

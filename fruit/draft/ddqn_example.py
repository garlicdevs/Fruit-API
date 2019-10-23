from fruit.agents.dqn import DQNAgent
from fruit.envs.ale import ALEEnvironment
from fruit.networks.policy import PolicyNetwork
from fruit.networks.config.atari import AtariDoubleDQNConfig
from stat import ST_CTIME
import os
import matplotlib.pyplot as plt
import numpy as np


def train_ale_environment():

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.PONG)
    ###################################################################################
    # In case using Gym
    # environment = GymEnvironment("Breakout-v0", state_processor=AtariProcessor())
    ###################################################################################

    # Create a network configuration for Atari DQN
    network_config = AtariDoubleDQNConfig(environment)

    # Create a policy network for DQN agent (create maximum of 40 checkpoints)
    network = PolicyNetwork(network_config, num_of_checkpoints=40)

    # Create DQN agent (Save checkpoint every 30 minutes, stop training at checkpoint 40th)
    agent = DQNAgent(network, environment, save_time_based=30, checkpoint_stop=40, log_dir="./train/ddqn_pong_time_based_30_40")

    # Train it
    agent.train()


def eval_ale_environment(model_path, render, num_of_epochs, steps_per_epoch, initial_epsilon, log_dir):

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT, is_render=render, max_episode_steps=5000)

    # Create a network configuration for Atari DQN
    network_config = AtariDoubleDQNConfig(environment)

    # Create a policy network for DQN agent
    network = PolicyNetwork(network_config, load_model_path=model_path)

    # Create DQN agent (Double DQN can use DQN agent)
    agent = DQNAgent(network, environment, initial_epsilon=initial_epsilon, report_frequency=1, num_of_threads=8,
                     num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir)

    # Evaluate it
    return agent.evaluate()


def performance_test(epsilon=0.02):

    log_dir = "./thi_test/ddqn_test/"
    model_path = "./train/ddqn_breakout_time_based_30_40_02-05-2018-14-50"
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

    # performance_test()

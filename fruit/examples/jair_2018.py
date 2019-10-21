from fruit.envs.games.food_collector.engine import FoodCollector
from fruit.envs.juice import FruitEnvironment
from fruit.utils.processor import AtariProcessor
from fruit.agents.factory import A3CAgent, MOA3CAgent, JairA3CAgent
from fruit.networks.policy import JairPolicyNetwork
from fruit.networks.config.atari import AtariA3CConfig, JairA3CConfig, Jair2A3CConfig
from fruit.envs.juice import RewardProcessor
from fruit.utils.processor import SeaquestProcessor
from fruit.agents.dqn import MODQNAgent
from fruit.envs.ale import ALEEnvironment
from fruit.networks.policy import PolicyNetwork
from fruit.networks.config.atari import AtariDQNConfig
import os.path
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import psutil


colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6', '#e04e39']
markers = ['s-', 'o-', '^-', 'v-', 'x-', 'h-']


class JairProcessor(RewardProcessor):
    def get_reward(self, rewards):
        return rewards[1]

    def clone(self):
        return JairProcessor()


def jair_train_a3c():

    game_engine = FoodCollector(render=False, speed=1000, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=False,
                           reward_processor=JairProcessor())

    network_config = AtariA3CConfig(env, initial_learning_rate=0.002)

    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True)

    agent = A3CAgent(network, env,
                     num_of_epochs=10,
                     steps_per_epoch=1e6,
                     save_frequency=5e5,
                     report_frequency=10,
                     update_network_frequency=4,
                     log_dir='./train/jair/food_collector/a3c_gpu_8_threads_epochs_10_lr_0002',
                     num_of_threads=8)

    agent.train()


def train_seaquest_environment():

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST,
                                 is_render=False,
                                 state_processor=SeaquestProcessor(),
                                 loss_of_life_termination=True,
                                 multi_objs=True)

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, num_of_checkpoints=40, using_gpu=True)

    # Create A3C agent
    agent = A3CAgent(network, environment,
                     num_of_epochs=20,
                     steps_per_epoch=1e6,
                     save_frequency=5e5,
                     report_frequency=10,
                     update_network_frequency=5,
                     log_dir='./train/jair/seaquest/a3c_gpu_8_threads_epoches_20_lr_0004',
                     num_of_threads=8)

    # Train it
    agent.train()


def evaluate_seaquest_environment():

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST,
                                 is_render=True,
                                 state_processor=SeaquestProcessor(),
                                 loss_of_life_termination=True)

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True,
                            load_model_path='./train/jair/seaquest/a3c_gpu_8_threads_epoches_20_lr_0004_07-05-2018-11-35/model-5007677')

    # Create A3C agent
    agent = A3CAgent(network, environment,
                     num_of_epochs=20,
                     steps_per_epoch=1e6,
                     save_frequency=5e5,
                     report_frequency=10,
                     update_network_frequency=5,
                     log_dir='./test/jair/seaquest/a3c_gpu_8_threads_epoches_20_lr_0004',
                     num_of_threads=1)

    # Train it
    agent.evaluate()


def jair_evaluate_a3c():
    game_engine = FoodCollector(render=True, speed=60, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=False,
                           reward_processor=JairProcessor())

    network_config = AtariA3CConfig(env)

    network = PolicyNetwork(network_config,
                            using_gpu=True,
                            load_model_path='./train/jair/food_collector/a3c_gpu_8_threads_epochs_10_lr_0004_06-22-2018-23-01/model-5500367'
                            )

    agent = A3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=100000,
                     report_frequency=1,
                     log_dir='./test/jair/food_collector/a3c_gpu_8_threads_epoches_10_lr_0002',
                     num_of_threads=1)

    agent.evaluate()


def jair_train_seaquest_multiple_critics():

    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=False,
                         state_processor=SeaquestProcessor(),
                         loss_of_life_termination=True,
                         frame_skip=8,
                         multi_objs=True)

    network_config = JairA3CConfig(env,
                                   initial_learning_rate=0.004,
                                   num_of_objs=3,
                                   weights=[0.8, 0.1, 0.1])

    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True)

    agent = MOA3CAgent(network, env,
                       num_of_epochs=20,
                       steps_per_epoch=1e6,
                       save_frequency=5e5,
                       report_frequency=10,
                       update_network_frequency=10,
                       log_dir='./train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_08_01_01',
                       num_of_threads=8)

    agent.train()


def jair_eval_seaquest_multiple_critics(render=False, load_model_path='./train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_01_00_00_07-08-2018-05-16/model-19508602',
                               num_of_epochs=10, steps_per_epoch=1e4, weights=[1, 0, 0]):

    if render:
        is_render = True
        num_of_threads = 1
    else:
        is_render = False
        num_of_threads = 8

    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=is_render,
                         state_processor=SeaquestProcessor(),
                         loss_of_life_termination=False,
                         frame_skip=8,
                         multi_objs=True)

    network_config = JairA3CConfig(env,
                                   initial_learning_rate=0.004,
                                   num_of_objs=3,
                                   weights=weights)

    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True,
                            load_model_path=load_model_path)

    agent = MOA3CAgent(network, env,
                       num_of_epochs=num_of_epochs,
                       steps_per_epoch=steps_per_epoch,
                       save_frequency=5e5,
                       report_frequency=1,
                       update_network_frequency=10,
                       log_dir='./test/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001',
                       num_of_threads=num_of_threads)

    return agent.evaluate()


def jair_train_multiple_critics():

    game_engine = FoodCollector(render=False, speed=1000, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = JairA3CConfig(env, initial_learning_rate=0.001, num_of_objs=2, weights=[1, 0])

    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True)

    agent = MOA3CAgent(network, env,
                     num_of_epochs=20,
                     steps_per_epoch=1e6,
                     save_frequency=5e5,
                     report_frequency=10,
                     update_network_frequency=4,
                     log_dir='./train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_01_00',
                     num_of_threads=8)

    agent.train()


def jair_eval_multiple_critics(render=False, load_model_path='./train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_01_00_06-29-2018-16-34/model-9500961',
                               num_of_epochs=10, steps_per_epoch=1e4):

    if render:
        is_render = True
        speed = 8
        num_of_threads = 1
    else:
        is_render = False
        speed = 1000
        num_of_threads = 8

    game_engine = FoodCollector(render=is_render, speed=speed, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = JairA3CConfig(env, initial_learning_rate=0.001, num_of_objs=2, weights=[1, 0])

    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            load_model_path=load_model_path,
                            using_gpu=True)

    agent = MOA3CAgent(network, env,
                       num_of_epochs=num_of_epochs,
                       steps_per_epoch=steps_per_epoch,
                       save_frequency=5e5,
                       report_frequency=10,
                       update_network_frequency=4,
                       log_dir='./test/jair/food_collector/multiple_critics/a3c_gpu_20_threads_epochs_10_lr_0001_01_00',
                       num_of_threads=num_of_threads)

    return agent.evaluate()


# Multiple policies
def jair_train_single_critic():
    game_engine = FoodCollector(render=False, speed=1000, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    # 2 weights [0,1] [1,0] -> 678 minutes -> CPU 45%
    # 3 weights [0,1] [1, 0] [0.5,0.5] -> 686 minutes
    # 4 weights [0,1] [0.33, 0.67] [0.67, 0.33] [1, 0] -> 690 minutes
    # 5 weights
    network_config = Jair2A3CConfig(env, initial_learning_rate=0.001, num_of_objs=2, weights=[[0, 1], [0.33, 0.67], [0.5, 0.5], [0.67, 0.33], [1, 0]])

    network = JairPolicyNetwork(network_config,
                                num_of_checkpoints=40,
                                using_gpu=True)

    agent = JairA3CAgent(network, env,
                         num_of_epochs=10,
                         steps_per_epoch=1e6,
                         save_frequency=5e5,
                         report_frequency=10,
                         update_network_frequency=4,
                         log_dir='./train/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_5_weights',
                         num_of_threads=20)

    agent.train()


def jair_eval_single_critics(weights, render=False, load_model_path='./train/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_2_weights_09-20-2018-16-06/model-9502807',
                            num_of_epochs=10, steps_per_epoch=1e4):

    if render:
        is_render = True
        speed = 8
        num_of_threads = 1
    else:
        is_render = False
        speed = 1000
        num_of_threads = 8

    game_engine = FoodCollector(render=is_render, speed=speed, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = Jair2A3CConfig(env, initial_learning_rate=0.001, num_of_objs=2,
                                    weights=weights)

    network = JairPolicyNetwork(network_config,
                                num_of_checkpoints=40,
                                load_model_path=load_model_path,
                                using_gpu=True)

    agent = JairA3CAgent(network, env,
                         num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch,
                         save_frequency=5e5,
                         report_frequency=10,
                         update_network_frequency=4,
                         log_dir='./test/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_5_weights',
                         num_of_threads=num_of_threads)

    return agent.evaluate()


def jair_train_seaquest_single_critic():
    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=False,
                         state_processor=SeaquestProcessor(),
                         loss_of_life_termination=True,
                         frame_skip=8,
                         multi_objs=True)

    # 3 Weights: weights=[[0, 0.5, 0.5], [0.4, 0.3, 0.3], [1, 0, 0]]
    # 4 Weights: weights=[[0, 0.5, 0.5], [0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [1, 0, 0]]
    # 5 weights: weights=[[0, 0.5, 0.5], [0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [0.6, 0.2, 0.2], [1, 0, 0]]
    network_config = Jair2A3CConfig(env, initial_learning_rate=0.004, num_of_objs=3, weights=[[0, 0.5, 0.5], [0.2, 0.4, 0.4],  [0.4, 0.3, 0.3], [0.6, 0.2, 0.2], [0.8, 0.1, 0.1], [1, 0, 0]])

    network = JairPolicyNetwork(network_config,
                                num_of_checkpoints=40,
                                using_gpu=False)

    agent = JairA3CAgent(network, env,
                         num_of_epochs=20,
                         steps_per_epoch=1e6,
                         save_frequency=5e5,
                         report_frequency=10,
                         update_network_frequency=10,
                         log_dir='./train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_6_weights',
                         num_of_threads=24)

    agent.train()


def jair_eval_seaquest_single_critics(weights, render=False, load_model_path='./train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_3_weights_07-11-2018-12-27/model-19513692',
                                      num_of_epochs=10, steps_per_epoch=1e4):

    if render:
        is_render = True
        num_of_threads = 1
    else:
        is_render = False
        num_of_threads = 8

    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=is_render,
                         state_processor=SeaquestProcessor(),
                         loss_of_life_termination=False,
                         frame_skip=8,
                         multi_objs=True)

    network_config = Jair2A3CConfig(env, initial_learning_rate=0.004, num_of_objs=3,
                                    weights=weights)

    network = JairPolicyNetwork(network_config,
                                num_of_checkpoints=40,
                                load_model_path=load_model_path,
                                using_gpu=True)

    agent = JairA3CAgent(network, env,
                         num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch,
                         save_frequency=5e5,
                         report_frequency=10,
                         update_network_frequency=10,
                         log_dir='./train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_6_weights',
                         num_of_threads=num_of_threads)

    return agent.evaluate()


def icmlsc_train_seaquest_mutiple_replays():
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=False,
                         state_processor=SeaquestProcessor(),
                         loss_of_life_termination=True,
                         frame_skip=8,
                         multi_objs=True)

    # Create a network configuration for Atari DQN
    network_config = AtariDQNConfig(environment)

    # Create a policy network for DQN agent (create maximum of 40 checkpoints)
    network = PolicyNetwork(network_config, num_of_checkpoints=40)

    # Create DQN agent (Save checkpoint every 30 minutes, stop training at checkpoint 40th)
    agent = MODQNAgent(network, environment, num_of_epochs=20, steps_per_epoch=1e6,
                       save_frequency=5e5,
                       epsilon_anneal_steps=2e6,
                       log_dir="./train/jair/modqn_1_thread/dqn_thread_9_modqn_3_weights_20",
                       exp_replay_size=600000,
                       num_of_threads=9,
                       weights=[[1, 0, 0], [0.4, 0.3, 0.3], [0, 0.5, 0.5]])

    # Train it
    agent.train()


def icmlsc_train_seaquest_single_replay():
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST,
                                 is_render=False,
                                 state_processor=SeaquestProcessor(),
                                 loss_of_life_termination=True,
                                 frame_skip=8,
                                 multi_objs=True)

    # Create a network configuration for Atari DQN
    network_config = AtariDQNConfig(environment)

    # Create a policy network for DQN agent (create maximum of 40 checkpoints)
    network = PolicyNetwork(network_config, num_of_checkpoints=40)

    # Create DQN agent (Save checkpoint every 30 minutes, stop training at checkpoint 40th)
    agent = MODQNAgent(network, environment, num_of_epochs=10, steps_per_epoch=1e6,
                       save_frequency=5e5,
                       log_dir="./train/jair/modqn_1_thread/dqn_thread_1_modqn_1_weight",
                       exp_replay_size=500000,
                       weights=[[1, 0, 0]])

    # Train it
    agent.train()


def find_models(path):
    numbers = []
    for name in os.listdir(path):
        if name.lower().endswith('.index'):
            number_cut = name[6:-6]
            numbers.append(int(number_cut))
    numbers = sorted(numbers)
    print(numbers)
    return [str(x) for x in numbers]


def jair_eval_scmp_mcsp():
    total_scores_1 = []
    result_dir = './train/jair/food_collector/multiple_critics/results/w_05_05.p'
    print("File for weight [0.5, 0.5] exists! Skip!")
    file = open(result_dir, 'rb+')
    total_scores_1 = pickle.load(file)
    print(total_scores_1)

    # Food Collector with 4 weights
    total_scores_2 = []
    result_dir = './train/jair/food_collector/single_critic/results/w_4w' + '_' + str(0.5) + '_' + str(0.5) + '.p'
    print("File for 4 weights exists! Skip!")
    file = open(result_dir, 'rb+')
    total_scores_2 = pickle.load(file)
    print(total_scores_2)

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    scale_factor = 2.
    y_ranges = [0.5 * i for i in range(1, 20)]
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='MCSP - (0.5, 0.5)')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='SCMP - $P_4$ - (0.5, 0.5)')


    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Number of Steps per Episode (Efficiency)')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0.5, 9.5)
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    axes.set_xticks(r)
    plt.ylim(-10, 280)
    axes.set_yticks([0, 100, 200, 300, 400, 500, 600])
    plt.savefig('./train/jair/food_collector/single_critic/results/scmp_mcsp.pdf')
    plt.show()


def eval_mcsp_food_collector():
    steps_per_epoch = 1e4
    num_of_epoches = 5
    total_scores = []

    # Food Collector with weight [1, 0]
    result_dir = './train/jair/food_collector/multiple_critics/results/w_1_0.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_01_00_06-29-2018-16-34'
        models = ['500066','1000098', '1500195', '2000216', '2500226', '3000278', '3500299', '4000401', '4500454', '5000526',
                  '5500605', '6000631', '6500678', '7000731', '7500824', '8000867', '8500881', '9000891', '9500961']

        for m in models:
            full_model_path = model_path + '/model-' + m

            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches, steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores, file)
    else:
        print("File for weight [1, 0] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores = pickle.load(file)
        print(total_scores)

    # Food Collector with weight [0.2, 0.8]
    total_scores_1 = []
    result_dir = './train/jair/food_collector/multiple_critics/results/w_033_067.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_02_08_06-28-2018-22-57'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)
            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_1.append(score_dist)
        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_1, file)
    else:
        print("File for weight [0.2, 0.8] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_1 = pickle.load(file)
        print(total_scores_1)

    # Food Collector with weight [0.5, 0.5]
    total_scores_2 = []
    result_dir = './train/jair/food_collector/multiple_critics/results/w_05_05.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_05_05_06-28-2018-03-33'
        models = ['500092', '1000280', '1500439', '2000560', '2500733', '3000804', '3500899', '4000913', '4501054',
                  '5001204',
                  '5501407', '6001432', '6501659', '7001888', '7501933', '8002144', '8502346', '9002397', '9502467']
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_2.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_2, file)
    else:
        print("File for weight [0.5, 0.5] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_2 = pickle.load(file)
        print(total_scores_2)

    # Food Collector with weight [0.8, 0.2]
    total_scores_3 = []
    result_dir = './train/jair/food_collector/multiple_critics/results/w_067_033.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_08_02_06-29-2018-16-33'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_3.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_3, file)
    else:
        print("File for weight [0.8, 0.2] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_3 = pickle.load(file)
        print(total_scores_3)

    # Food Collector with weight [0, 1]
    total_scores_4 = []
    result_dir = './train/jair/food_collector/multiple_critics/results/w_0_1.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_00_01_06-30-2018-20-20'
        models = ['500099', '1000288', '1500402', '2000457', '2500634', '3000769', '3501004', '4001103', '4501216',
                  '5001385',
                  '5501391', '6001424', '6501614', '7001840', '7501939', '8001958', '8502089', '9002127',
                  '9502171']
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_4.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_4, file)
    else:
        print("File for weight [0, 1] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_4 = pickle.load(file)
        print(total_scores_4)

    # Plot the result (first score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = [0.5*i for i in range(1, 20)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i-1] + mean_scores[i])/2
        error_scores.append(8*stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores), np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='w = 1.0, 0.0')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='w = 0.67, 0.33')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='w = 0.5, 0.5')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='w = 0.33, 0.67')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_4)):
        temp_scores = []
        for j in range(len(total_scores_4[i])):
            temp_scores.append(total_scores_4[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)

    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[5], edgecolor=colors[5], label='w = 0.0, 1.0')
    plt.grid(True)
    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Total Rewards (Collecting Food)')
    plt.legend(loc='best')
    #plt.title('Performance of MCSP with Different Weights in Food Collector')
    axes = plt.gca()
    axes.set_xlim(0.5, 9.5)
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    axes.set_xticks(r)
    plt.ylim(-10, 525)
    axes.set_yticks([0, 100, 200, 300, 400, 500])
    plt.savefig('./train/jair/food_collector/multiple_critics/results/perform1.pdf')
    plt.show()

    # Plot the result (second score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = [0.5 * i for i in range(1, 20)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='w = 1.0, 0.0')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='w = 0.67, 0.33')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='w = 0.5, 0.5')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='w = 0.33, 0.67')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_4)):
        temp_scores = []
        for j in range(len(total_scores_4[i])):
            temp_scores.append(total_scores_4[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[5], edgecolor=colors[5], label='w = 0.0, 1.0')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Auxiliary Rewards (Eating Food)')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0.5, 9.5)
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    axes.set_xticks(r)
    plt.ylim(-10, 280)
    axes.set_yticks([0, 50, 100, 150, 200, 250])
    plt.savefig('./train/jair/food_collector/multiple_critics/results/perform2.pdf')
    plt.show()


def eval_scmp_food_collector(weights=[[0.5, 0.5]]):
    steps_per_epoch = 1e4
    num_of_epoches = 5
    total_scores = []
    p1 = weights[0][0]
    p2 = weights[0][1]

    result_dir = './train/jair/food_collector/multiple_critics/results/w_1_0.p'
    file = open(result_dir, 'rb+')
    total_scores_base = pickle.load(file)
    print(total_scores_base)

    # Food Collector with 2 weights
    if p1 == 1 and p2 == 0:
        result_dir = './train/jair/food_collector/single_critic/results/w_2w.p'
    else:
        result_dir = './train/jair/food_collector/single_critic/results/w_2w' + '_' + str(p1) + '_' + str(p2) + '.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_2_weights_09-20-2018-16-06'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            score_dist = jair_eval_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches, steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores, file)
    else:
        print("File for 2 weights exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores = pickle.load(file)
        print(total_scores)

    # Food Collector with 3 weights
    total_scores_1 = []
    if p1 == 1 and p2 == 0:
        result_dir = './train/jair/food_collector/single_critic/results/w_3w.p'
    else:
        result_dir = './train/jair/food_collector/single_critic/results/w_3w' + '_' + str(p1) + '_' + str(p2) + '.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_3_weights_09-21-2018-14-28'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)
            score_dist = jair_eval_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                  steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_1.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_1, file)
    else:
        print("File for 3 weights exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_1 = pickle.load(file)
        print(total_scores_1)

    # Food Collector with 4 weights
    total_scores_2 = []
    if p1 == 1 and p2 == 0:
        result_dir = './train/jair/food_collector/single_critic/results/w_4w.p'
    else:
        result_dir = './train/jair/food_collector/single_critic/results/w_4w' + '_' + str(p1) + '_' + str(p2) + '.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_4_weights_07-12-2018-02-35'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)
            score_dist = jair_eval_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                  steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_2.append(score_dist)
        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_2, file)
    else:
        print("File for 4 weights exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_2 = pickle.load(file)
        print(total_scores_2)

    # Food Collector with 5 weights
    total_scores_3 = []
    if p1 == 1 and p2 == 0:
        result_dir = './train/jair/food_collector/single_critic/results/w_5w.p'
    else:
        result_dir = './train/jair/food_collector/single_critic/results/w_5w' + '_' + str(p1) + '_' + str(p2) + '.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_5_weights_07-12-2018-02-37'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)
            score_dist = jair_eval_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_3.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_3, file)
    else:
        print("File for 5 weights exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_3 = pickle.load(file)
        print(total_scores_3)

    # Plot the result
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    scale_factor = 2
    y_ranges = [0.5*i for i in range(1, 20)]

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_base)):
        temp_scores = []
        for j in range(len(total_scores_base[i])):
            temp_scores.append(total_scores_base[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='A3C Baseline')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i-1] + mean_scores[i])/2
        error_scores.append(scale_factor*stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores), np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[1], edgecolor=colors[1], label='SCMP - $P_2$')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='SCMP - $P_3$')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='SCMP - $P_4$')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='SCMP - $P_5$')

    plt.grid(True)
    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Total Rewards (Collecting Food)')
    plt.legend(loc='best')
    #plt.title('Performance of SCMP with Different Weight Sets in Food Collector')
    axes = plt.gca()
    axes.set_xlim(0.5, 9.5)
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    axes.set_xticks(r)
    plt.ylim(-10, 525)
    axes.set_yticks([0, 100, 200, 300, 400, 500])
    plt.savefig('./train/jair/food_collector/single_critic/results/perform1_'+str(p1)+'_'+str(p2)+'.pdf')
    plt.show()

    # Plot the result (second score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_base)):
        temp_scores = []
        for j in range(len(total_scores_base[i])):
            temp_scores.append(total_scores_base[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='A3C Baseline')

    y_ranges = [0.5 * i for i in range(1, 20)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[1], edgecolor=colors[1], label='SCMP - $P_2$')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='SCMP - $P_3$')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='SCMP - $P_4$')

    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='SCMP - $P_5$')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Auxiliary Rewards (Eating Food)')
    plt.legend(loc='best')
    #plt.title('Performance of SCMP with Different Weight Sets in Food Collector')
    axes = plt.gca()
    axes.set_xlim(0.5, 9.5)
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    axes.set_xticks(r)
    plt.ylim(-10, 280)
    axes.set_yticks([0, 50, 100, 150, 200, 250])
    plt.savefig('./train/jair/food_collector/single_critic/results/perform2_' + str(p1) + '_' + str(p2) + '.pdf')
    plt.show()


def smooth(stat, times=1):
    for j in range(times):
        for i in range(len(stat)-1):
            vol = stat[i]
            next_vol = stat[i+1]
            vol = (vol + next_vol)/2
            stat[i] = vol
    return stat


def eval_mcsp_seaquest():
    steps_per_epoch = 1e4
    num_of_epoches = 5
    total_scores = []

    # Seaquest with weight [1, 0]
    result_dir = './train/jair/seaquest/multiple_critics/results/w_1_0_0.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_01_00_00_07-08-2018-05-16'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m

            score_dist = jair_eval_seaquest_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches, steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores, file)
    else:
        print("File for weight [1, 0, 0] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores = pickle.load(file)
        print(total_scores)

    # Seaquest with weight [0.8, 0.1, 0.1]
    total_scores_1 = []
    result_dir = './train/jair/seaquest/multiple_critics/results/w_08_01_01.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_08_01_01_07-08-2018-22-50'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_seaquest_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches, steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_1.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_1, file)
    else:
        print("File for weight [0.8, 0.1, 0.1] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_1 = pickle.load(file)
        print(total_scores_1)

    # Seaquest with weight [0.6, 0.2, 0.2]
    total_scores_2 = []
    result_dir = './train/jair/seaquest/multiple_critics/results/w_06_02_02.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_06_02_02_07-08-2018-19-32'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_seaquest_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_2.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_2, file)
    else:
        print("File for weight [0.6, 0.2, 0.2] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_2 = pickle.load(file)
        print(total_scores_2)

    # Seaquest with weight [0.4, 0.3, 0.3]
    total_scores_3 = []
    result_dir = './train/jair/seaquest/multiple_critics/results/w_04_03_03.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_04_03_03_07-08-2018-05-17'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_seaquest_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_3.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_3, file)
    else:
        print("File for weight [0.4, 0.3, 0.3] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_3 = pickle.load(file)
        print(total_scores_3)

    # Seaquest with weight [0.2, 0.4, 0.4]
    total_scores_4 = []
    result_dir = './train/jair/seaquest/multiple_critics/results/w_02_04_04.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_02_04_04_07-08-2018-05-16'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_seaquest_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_4.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_4, file)
    else:
        print("File for weight [0.2, 0.4, 0.4] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_4 = pickle.load(file)
        print(total_scores_4)

    # Seaquest with weight [0.0, 0.5, 0.5]
    total_scores_5 = []
    result_dir = './train/jair/seaquest/multiple_critics/results/w_00_05_05.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_00_05_05_07-08-2018-19-31'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)
            score_dist = jair_eval_seaquest_multiple_critics(load_model_path=full_model_path,
                                                             num_of_epochs=num_of_epoches,
                                                             steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_5.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_5, file)
    else:
        print("File for weight [0.0, 0.5, 0.5] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_5 = pickle.load(file)
        print(total_scores_5)

    # Plot the result (first score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    scale_factor = 2
    y_ranges = [0.5*i for i in range(1, 40)]
    mean_scores = []
    error_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i-1] + mean_scores[i])/2
        error_scores.append(scale_factor*stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores), np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='w = 1.0, 0.0, 0.0')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='w = 0.8, 0.1, 0.1')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='w = 0.6, 0.2, 0.2')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='w = 0.4, 0.3, 0.3')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_4)):
        temp_scores = []
        for j in range(len(total_scores_4[i])):
            temp_scores.append(total_scores_4[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[5], edgecolor=colors[5], label='w = 0.2, 0.4, 0.4')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_5)):
        temp_scores = []
        for j in range(len(total_scores_5[i])):
            temp_scores.append(total_scores_5[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[7])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[7], edgecolor=colors[7], label='w = 0.0, 0.5, 0.5')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Total Rewards')
    plt.legend(loc='best')
    #plt.title('Performance of MCSP with Different Weights in Seaquest')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-10, 4600)
    axes.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
    plt.savefig('./train/jair/seaquest/multiple_critics/results/perform1.pdf')
    plt.show()

    # Plot the result (second score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    scale_factor = 2
    y_ranges = [0.5*i for i in range(1, 40)]
    mean_scores = []
    error_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i-1] + mean_scores[i])/2
        error_scores.append(scale_factor*stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores), np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='w = 1.0, 0.0, 0.0')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='w = 0.8, 0.1, 0.1')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='w = 0.6, 0.2, 0.2')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='w = 0.4, 0.3, 0.3')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_4)):
        temp_scores = []
        for j in range(len(total_scores_4[i])):
            temp_scores.append(total_scores_4[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[5], edgecolor=colors[5], label='w = 0.2, 0.4, 0.4')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_5)):
        temp_scores = []
        for j in range(len(total_scores_5[i])):
            temp_scores.append(total_scores_5[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[7])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[7], edgecolor=colors[7], label='w = 0.0, 0.5, 0.5')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Auxiliary Rewards (Rescuing human)')
    plt.legend(loc='best')
    #plt.title('Performance of MCSP with Different Weights in Seaquest')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-10, 310)
    axes.set_yticks([0, 50, 100, 150, 200, 250, 300])
    plt.savefig('./train/jair/seaquest/multiple_critics/results/perform2.pdf')
    plt.show()

    # Plot the result (third score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    scale_factor = 2
    y_ranges = [0.5*i for i in range(1, 40)]
    mean_scores = []
    error_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i-1] + mean_scores[i])/2
        error_scores.append(scale_factor*stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores), np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='w = 1.0, 0.0, 0.0')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='w = 0.8, 0.1, 0.1')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='w = 0.6, 0.2, 0.2')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='w = 0.4, 0.3, 0.3')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_4)):
        temp_scores = []
        for j in range(len(total_scores_4[i])):
            temp_scores.append(total_scores_4[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[5], edgecolor=colors[5], label='w = 0.2, 0.4, 0.4')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_5)):
        temp_scores = []
        for j in range(len(total_scores_5[i])):
            temp_scores.append(total_scores_5[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[7])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[7], edgecolor=colors[7], label='w = 0.0, 0.5, 0.5')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Auxiliary Rewards (Resurfacing)')
    plt.legend(loc='best')
    #plt.title('Performance of MCSP with Different Weights in Seaquest')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-50, 2200)
    axes.set_yticks([0, 500, 1000, 1500, 2000])
    plt.savefig('./train/jair/seaquest/multiple_critics/results/perform3.pdf')
    plt.show()


def eval_scmp_seaquest(weights = [[0.4, 0.3, 0.3]]):
    steps_per_epoch = 1e4
    num_of_epoches = 5
    total_scores = []

    result_dir = './train/jair/seaquest/multiple_critics/results/w_1_0_0.p'
    file = open(result_dir, 'rb+')
    total_scores_base = pickle.load(file)
    print(total_scores_base)

    p1 = weights[0][0]
    p2 = weights[0][1]
    p3 = weights[0][2]

    # Seaquest with 3 weights
    if p1 == 1 and p2 == 0 and p3 == 0:
        result_dir = './train/jair/seaquest/single_critic/results/w_3w.p'
    else:
        result_dir = './train/jair/seaquest/single_critic/results/w_3w'+str(p1)+'_'+str(p2)+'_'+str(p3)+'.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_3_weights_07-11-2018-12-27'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            score_dist = jair_eval_seaquest_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches, steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores, file)
    else:
        print("File for 3 weights exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores = pickle.load(file)
        print(total_scores)

    # Seaquest with 4 weights
    total_scores_1 = []
    if p1 == 1 and p2 == 0 and p3 == 0:
        result_dir = './train/jair/seaquest/single_critic/results/w_4w.p'
    else:
        result_dir = './train/jair/seaquest/single_critic/results/w_4w'+str(p1)+'_'+str(p2)+'_'+str(p3)+'.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_4_weights_07-10-2018-23-24'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)
            score_dist = jair_eval_seaquest_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches, steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_1.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_1, file)
    else:
        print("File for 4 weights exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_1 = pickle.load(file)
        print(total_scores_1)

    # Seaquest with 5 weights
    total_scores_2 = []
    if p1 == 1 and p2 == 0 and p3 == 0:
        result_dir = './train/jair/seaquest/single_critic/results/w_5w.p'
    else:
        result_dir = './train/jair/seaquest/single_critic/results/w_5w'+str(p1)+'_'+str(p2)+'_'+str(p3)+'.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_5_weights_07-11-2018-11-24'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_seaquest_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_2.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_2, file)
    else:
        print("File for weight [0.6, 0.2, 0.2] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_2 = pickle.load(file)
        print(total_scores_2)

    # Seaquest with 6 weights
    total_scores_3 = []
    if p1 == 1 and p2 == 0 and p3 == 0:
        result_dir = './train/jair/seaquest/single_critic/results/w_6w.p'
    else:
        result_dir = './train/jair/seaquest/single_critic/results/w_6w'+str(p1)+'_'+str(p2)+'_'+str(p3)+'.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_6_weights_07-11-2018-17-40'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m
            print(full_model_path)

            score_dist = jair_eval_seaquest_single_critics(weights=weights, load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_3.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_3, file)
    else:
        print("File for weight [0.4, 0.3, 0.3] exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_3 = pickle.load(file)
        print(total_scores_3)

    # Plot the result (first score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    scale_factor = 2
    y_ranges = [0.5*i for i in range(1, 40)]

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_base)):
        temp_scores = []
        for j in range(len(total_scores_base[i])):
            temp_scores.append(total_scores_base[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='A3C Baseline')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i-1] + mean_scores[i])/2
        error_scores.append(scale_factor*stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores), np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[1], edgecolor=colors[1], label='SCMP - $P_3$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='SCMP - $P_4$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='SCMP - $P_5$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='SCMP - $P_6$')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Total Rewards')
    plt.legend(loc='best')
    #plt.title('Performance of SCMP with Different Weight Sets in Seaquest')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-10, 4600)
    axes.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
    plt.savefig('./train/jair/seaquest/single_critic/results/perform1_'+ str(p1) + '_' + str(p2) + '_' + str(p3) + '.pdf')
    plt.show()

    # Plot the result (2nd score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    scale_factor = 2
    y_ranges = [0.5 * i for i in range(1, 40)]

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_base)):
        temp_scores = []
        for j in range(len(total_scores_base[i])):
            temp_scores.append(total_scores_base[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='A3C Baseline')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[1], edgecolor=colors[1], label='SCMP - $P_3$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='SCMP - $P_4$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='SCMP - $P_5$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='SCMP - $P_6$')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Auxiliary Rewards (Rescuing Human)')
    plt.legend(loc='best')
    #plt.title('Performance of SCMP with Different Weight Sets in Seaquest')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-10, 310)
    axes.set_yticks([0, 50, 100, 150, 200, 250, 300])
    plt.savefig('./train/jair/seaquest/single_critic/results/perform2_'+ str(p1) + '_' + str(p2) + '_' + str(p3) + '.pdf')
    plt.show()

    # Plot the result (3rd score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    scale_factor = 2
    y_ranges = [0.5 * i for i in range(1, 40)]

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_base)):
        temp_scores = []
        for j in range(len(total_scores_base[i])):
            temp_scores.append(total_scores_base[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[0])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[0], edgecolor=colors[0], label='A3C Baseline')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[1], edgecolor=colors[1], label='SCMP - $P_3$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='SCMP - $P_4$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[4], edgecolor=colors[4], label='SCMP - $P_5$')

    mean_scores = []
    error_scores = []
    for i in range(len(total_scores_3)):
        temp_scores = []
        for j in range(len(total_scores_3[i])):
            temp_scores.append(total_scores_3[i][j][0][2])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(scale_factor * stats.sem(temp_scores))
    print(mean_scores)
    print(error_scores)
    mean_scores = smooth(mean_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='SCMP - $P_6$')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Auxiliary Rewards (Resurfacing)')
    plt.legend(loc='best')
    #plt.title('Performance of SCMP with Different Weight Sets in Seaquest')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-50, 2200)
    axes.set_yticks([0, 500, 1000, 1500, 2000])
    plt.savefig('./train/jair/seaquest/single_critic/results/perform3_'+ str(p1) + '_' + str(p2) + '_' + str(p3) + '.pdf')
    plt.show()


def eval_a3c_food_collector():
    steps_per_epoch = 1e4
    num_of_epoches = 5

    # A3C - 10 epoches
    total_scores = []
    result_dir = './train/jair/food_collector/multiple_critics/results/w_1_0.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_00_01_06-30-2018-20-20'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m

            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores, file)
    else:
        print("File for A3C-10 exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores = pickle.load(file)
        print(total_scores)

    # A3C - 20 epoches
    total_scores_1 = []
    result_dir = './train/jair/food_collector/multiple_critics/results/a3c_org_20.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_01_00_09-23-2018-13-05'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m

            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_1.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_1, file)
    else:
        print("File for A3C-10 exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_1 = pickle.load(file)
        print(total_scores_1)

    # A3C - 20 threads - 10 epoches
    total_scores_2 = []
    result_dir = './train/jair/food_collector/multiple_critics/results/a3c_org_20t_10.p'
    if not os.path.isfile(result_dir):
        model_path = './train/jair/food_collector/multiple_critics/a3c_gpu_20_threads_epochs_10_lr_0001_01_00_07-19-2018-10-33'
        models = find_models(model_path)
        for m in models:
            full_model_path = model_path + '/model-' + m

            score_dist = jair_eval_multiple_critics(load_model_path=full_model_path, num_of_epochs=num_of_epoches,
                                                    steps_per_epoch=steps_per_epoch)
            print(score_dist)
            total_scores_2.append(score_dist)

        file = open(result_dir, 'wb+')
        pickle.dump(total_scores_2, file)
    else:
        print("File for A3C-10 exists! Skip!")
        file = open(result_dir, 'rb+')
        total_scores_2 = pickle.load(file)
        print(total_scores_2)

    # Plot the result (first score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    #plt.axvspan(0.5, 9.5, facecolor=colors[5], alpha=0.02)
    #plt.axvspan(9.5, 19.5, facecolor=colors[8], alpha=0.02)
    ys = [-10 + 10*i for i in range(54)]
    xs = [9.5 for _ in range(54)]
    plt.grid(True)
    y_ranges = [0.5 * i for i in range(1, 20)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[1], edgecolor=colors[1], label='A3C - 8 threads - 10 million training steps')

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    y_ranges = [0.5 * i for i in range(1, 40)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2], label='A3C - 8 threads - 20 million training steps')

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    y_ranges = [0.5 * i for i in range(1, 20)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][0])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3], label='A3C - 20 threads - 10 million training steps')
    plt.plot(xs, ys, alpha=0.2, color='gray')
    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Total Rewards (Collecting Food)')
    plt.legend(loc='best')
    #plt.title('Performance of A3C in Food Collector')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-10, 525)
    axes.set_yticks([0, 100, 200, 300, 400, 500])
    plt.savefig('./train/jair/food_collector/multiple_critics/results/a3c_1.pdf')
    plt.show()

    # Plot the result (2nd score)
    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    #plt.axvspan(0.5, 9.5, facecolor=colors[5], alpha=0.02)
    #plt.axvspan(9.5, 19.5, facecolor=colors[8], alpha=0.02)
    ys = [-50 + 10 * i for i in range(54)]
    xs = [9.5 for _ in range(54)]
    plt.grid(True)
    y_ranges = [0.5 * i for i in range(1, 20)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores)):
        temp_scores = []
        for j in range(len(total_scores[i])):
            temp_scores.append(total_scores[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[1], edgecolor=colors[1],
                     label='A3C - 8 threads - 10 million training steps')

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    y_ranges = [0.5 * i for i in range(1, 40)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_1)):
        temp_scores = []
        for j in range(len(total_scores_1[i])):
            temp_scores.append(total_scores_1[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[2])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[2], edgecolor=colors[2],
                     label='A3C - 8 threads - 20 million training steps')

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)
    y_ranges = [0.5 * i for i in range(1, 20)]
    mean_scores = []
    error_scores = []
    min_scores = []
    max_scores = []
    for i in range(len(total_scores_2)):
        temp_scores = []
        for j in range(len(total_scores_2[i])):
            temp_scores.append(total_scores_2[i][j][0][1])
        print(stats.sem(temp_scores))
        mean_scores.append(np.mean(temp_scores))
        if mean_scores[i] == 0:
            mean_scores[i] = (mean_scores[i - 1] + mean_scores[i]) / 2
        error_scores.append(8 * stats.sem(temp_scores))
        min_scores.append(np.min(temp_scores))
        max_scores.append(np.max(temp_scores))
    print(mean_scores)
    print(error_scores)
    plt.plot(y_ranges, mean_scores, alpha=0.8, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(error_scores),
                     np.array(mean_scores) + np.array(error_scores),
                     alpha=0.2, facecolor=colors[3], edgecolor=colors[3],
                     label='A3C - 20 threads - 10 million training steps')
    plt.plot(xs, ys, alpha=0.2, color='gray')
    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Auxiliary Rewards (Eating Food)')
    plt.legend(loc='upper right')
    #plt.title('Performance of A3C in Food Collector')
    axes = plt.gca()
    axes.set_xlim(0.5, 19.5)
    r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    axes.set_xticks(r)
    plt.ylim(-10, 280)
    axes.set_yticks([0, 50, 100, 150, 200, 250])
    plt.savefig('./train/jair/food_collector/multiple_critics/results/a3c_2.pdf')
    plt.show()


if __name__ == "__main__":
    # jair_train_multiple_critics()

    # jair_train_seaquest_multiple_critics()

    # evaluate_seaquest_environment()

    # jair_eval_seaquest_multiple_critics()

    # jair_train_single_critic()

    # jair_train_seaquest_single_critic()

    # jair_train_multiple_critics()

    # icmlsc_train_seaquest_single_replay()

    # icmlsc_train_seaquest_mutiple_replays()

    # jair_eval_multiple_critics()

    # jair_train_single_critic()

    # jair_train_multiple_critics()

    # eval_mcsp_food_collector()

    # eval_mcsp_seaquest()

    # eval_a3c_food_collector()

    # eval_scmp_food_collector(weights=[[0, 1]])

    # eval_scmp_seaquest(weights=[[0, 0.5, 0.5]])

    #jair_eval_multiple_critics(render=True, load_model_path="./train/jair/food_collector/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001_05_05_06-28-2018-03-33/model-9502467")

    # jair_eval_single_critics(render=True, weights=[[0.5, 0.5]], load_model_path="./train/jair/food_collector/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_5_weights_07-12-2018-02-37/model-9501532")

    # jair_eval_seaquest_multiple_critics(render=True, load_model_path='/home/garlicdevs/Dropbox/PhD/Tensorflow/A3C/fruit/examples/train/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_w_04_03_03_07-08-2018-05-17/model-19509950',
    #                                    weights=[0.4, 0.3, 0.3])

    # jair_eval_seaquest_single_critics(weights=[[0.4, 0.3, 0.3]], load_model_path='/home/garlicdevs/Dropbox/PhD/Tensorflow/A3C/fruit/examples/train/jair/seaquest/single_critic/a3c_gpu_8_threads_epochs_20_lr_0004_multiple_policy_256_4_weights_07-10-2018-23-24/model-19025013', render=True)

    # jair_eval_scmp_mcsp()

    eval_a3c_food_collector()
from fruit.configs.a3c import AtariA3CConfig
from fruit.envs.games.food_collector.engine import FoodCollector
from fruit.envs.juice import FruitEnvironment
from fruit.agents.factory import AgentFactory
from fruit.learners.a3c import A3CLearner
from fruit.networks.policy import JairPolicyNetwork
from fruit.networks.config.atari import JairA3CConfig, Jair2A3CConfig
from fruit.envs.juice import RewardProcessor
from fruit.envs.ale import ALEEnvironment
from fruit.networks.policy import PolicyNetwork
from fruit.state.advanced import SeaquestProcessor
from fruit.state.processor import AtariProcessor


class FCRewardProcessor(RewardProcessor):
    def get_reward(self, rewards):
        return rewards[1]

    def clone(self):
        return FCRewardProcessor()


def train_a3c_food_collector():
    game_engine = FoodCollector(render=False, speed=1000, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    environment = FruitEnvironment(game_engine, max_episode_steps=10000, state_processor=AtariProcessor(),
                                   reward_processor=FCRewardProcessor())

    network_config = AtariA3CConfig(environment, initial_learning_rate=0.002)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=10, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/food_collector_/a3c_checkpoints',
                                network_update_steps=4)

    agent.train()


def train_a3c_seaquest():

    environment = ALEEnvironment(ALEEnvironment.SEAQUEST, is_render=False, state_processor=SeaquestProcessor(),
                                 loss_of_life_termination=True)

    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=20, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/seaquest/a3c_checkpoints')

    agent.train()


def train_a3c_seaquest_multiple_critics():
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


def train_a3c_fc_multiple_critics():
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
                     log_dir='./train/jair/food_collector_/multiple_critics/a3c_gpu_8_threads_epochs_20_lr_0001_01_00',
                     num_of_threads=8)

    agent.train()


def train_a3c_fc_single_critic():
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
                         log_dir='./train/jair/food_collector_/single_critic/a3c_gpu_8_threads_epochs_10_lr_0001_multiple_policy_256_5_weights',
                         num_of_threads=20)

    agent.train()


def train_a3c_seaquest_single_critic():
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

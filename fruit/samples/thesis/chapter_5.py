from fruit.configs.a3c import AtariA3CConfig
from fruit.configs.a3c_multi_objectives import MOMCA3CConfig, MOSCA3CConfig
from fruit.envs.games.food_collector_.engine import FoodCollector
from fruit.envs.juice import FruitEnvironment
from fruit.agents.factory import AgentFactory
from fruit.learners.a3c import A3CLearner
from fruit.envs.juice import RewardProcessor
from fruit.envs.ale import ALEEnvironment
from fruit.learners.multi_objectives import MOSCA3CLearner
from fruit.networks.policy import PolicyNetwork
from fruit.state.advanced import SeaquestProcessor
from fruit.state.processor import AtariProcessor


class FCRewardProcessor(RewardProcessor):
    def get_reward(self, rewards):
        return rewards[0] # Collecting rewards

    def clone(self):
        return FCRewardProcessor()

    def get_number_of_objectives(self):
        return 1


def train_a3c_fc():
    game_engine = FoodCollector(render=False, speed=1000, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    environment = FruitEnvironment(game_engine, max_episode_steps=10000, state_processor=AtariProcessor(),
                                   reward_processor=FCRewardProcessor())

    network_config = AtariA3CConfig(environment, initial_learning_rate=0.002)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=20)

    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=10, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/food_collector/a3c_checkpoints',
                                network_update_steps=4)

    agent.train()


def train_a3c_sea_quest():
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST, is_render=False, state_processor=SeaquestProcessor(),
                                 loss_of_life_termination=True)

    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=20, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/seaquest/a3c_checkpoints')

    agent.train()


def train_a3c_sea_quest_multiple_critics():
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST, state_processor=SeaquestProcessor(),
                                 loss_of_life_termination=True, frame_skip=8)

    network_config = MOMCA3CConfig(environment, initial_learning_rate=0.004, weights=[0.8, 0.1, 0.1])

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=20, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/sea_quest/mo_mc_a3c_checkpoints',
                                network_update_steps=10)

    agent.train()


def train_a3c_fc_multiple_critics():
    game_engine = FoodCollector(render=False, speed=1000, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    environment = FruitEnvironment(game_engine, max_episode_steps=10000, state_processor=AtariProcessor())

    print('Number of objectives:', environment.get_number_of_objectives())

    network_config = MOMCA3CConfig(environment, initial_learning_rate=0.001, weights=[1, 0])

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=20, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/food_collector/mo_mc_a3c_checkpoints',
                                network_update_steps=4)

    agent.train()


def train_a3c_fc_single_critic():
    game_engine = FoodCollector(render=False, speed=1000, frame_skip=1, seed=None,
                                num_of_apples=1, human_control=False, debug=False)

    environment = FruitEnvironment(game_engine, max_episode_steps=10000, state_processor=AtariProcessor())

    # 2 weights [0,1] [1,0] -> 678 minutes -> CPU 45%
    # 3 weights [0,1] [1, 0] [0.5,0.5] -> 686 minutes
    # 4 weights [0,1] [0.33, 0.67] [0.67, 0.33] [1, 0] -> 690 minutes
    network_config = MOSCA3CConfig(environment, initial_learning_rate=0.001,
                                   weights=[[0, 1], [0.33, 0.67], [0.5, 0.5], [0.67, 0.33], [1, 0]])

    network = PolicyNetwork(network_config, max_num_of_checkpoints=20)

    agent = AgentFactory.create(MOSCA3CLearner, network, environment, num_of_epochs=10, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/food_collector/mo_sc_a3c_checkpoints',
                                network_update_steps=4, num_of_learners=20)

    agent.train()


def train_a3c_sea_quest_single_critic():
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST, state_processor=SeaquestProcessor(),
                                 loss_of_life_termination=True, frame_skip=8)

    # 3 Weights: weights=[[0, 0.5, 0.5], [0.4, 0.3, 0.3], [1, 0, 0]]
    # 4 Weights: weights=[[0, 0.5, 0.5], [0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [1, 0, 0]]
    # 5 weights: weights=[[0, 0.5, 0.5], [0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [0.6, 0.2, 0.2], [1, 0, 0]]
    network_config = MOSCA3CConfig(environment, initial_learning_rate=0.004,
                                   weights=[[0, 0.5, 0.5], [0.2, 0.4, 0.4],  [0.4, 0.3, 0.3],
                                            [0.6, 0.2, 0.2], [0.8, 0.1, 0.1], [1, 0, 0]])

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(MOSCA3CLearner, network, environment, num_of_epochs=20, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/sea_quest/mo_sc_a3c_checkpoints',
                                network_update_steps=10, num_of_learners=24)

    agent.train()


if __name__ == '__main__':
    train_a3c_sea_quest_single_critic()

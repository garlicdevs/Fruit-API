from fruit.agents.factory import AgentFactory
from fruit.configs.a3c_multi_agents import MAA3CConfig
from fruit.envs.games.milk_factory.engine import MilkFactory
from fruit.envs.juice import FruitEnvironment
from fruit.learners.multi_agents import MAA3CLearner
from fruit.networks.policy import PolicyNetwork
from fruit.state.processor import AtariProcessor


def train_milk_1_milk_1_fix_robots_with_no_status():
    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=1,
                              number_of_fix_robots=1, number_of_milks=1, seed=None, human_control=False,
                              error_freq=0.03, human_control_robot=0, milk_speed=3, debug=False,
                              action_combined_mode=False, show_status=False)

    environment = FruitEnvironment(game_engine, max_episode_steps=200, state_processor=AtariProcessor())

    network_config = MAA3CConfig(environment, initial_learning_rate=0.001, beta=0.001)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(MAA3CLearner, network, environment, num_of_epochs=40, steps_per_epoch=1e5,
                                checkpoint_frequency=1e5, log_dir='./train/milk_factory/a3c_ma_2_checkpoints')

    agent.train()


def train_milk_1_milk_1_fix_robots_with_status():
    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=1,
                              number_of_fix_robots=1, number_of_milks=1, seed=None, human_control=False,
                              error_freq=0.03, human_control_robot=0, milk_speed=3, debug=False,
                              action_combined_mode=False, show_status=True)

    environment = FruitEnvironment(game_engine, max_episode_steps=200, state_processor=AtariProcessor())

    network_config = MAA3CConfig(environment, initial_learning_rate=0.001, beta=0.001)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(MAA3CLearner, network, environment, num_of_epochs=40, steps_per_epoch=1e5,
                                checkpoint_frequency=1e5, log_dir='./train/milk_factory/a3c_ma_2_status_checkpoints')

    agent.train()


def train_milk_2_milk_1_fix_robots_with_no_status():
    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=2,
                              number_of_fix_robots=1, number_of_milks=2, seed=None, human_control=False, error_freq=0.01,
                              human_control_robot=0, milk_speed=3, debug=False, action_combined_mode=False, show_status=False,
                              number_of_exits=2)

    environment = FruitEnvironment(game_engine, max_episode_steps=200, state_processor=AtariProcessor())

    network_config = MAA3CConfig(environment, initial_learning_rate=0.001, beta=0.001)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(MAA3CLearner, network, environment, num_of_epochs=40, steps_per_epoch=1e5,
                                checkpoint_frequency=1e5, log_dir='./train/milk_factory/a3c_ma_3_checkpoints')

    agent.train()


def train_milk_2_milk_1_fix_robots_with_status():
    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=2,
                              number_of_fix_robots=1, number_of_milks=2, seed=None, human_control=False,
                              error_freq=0.01, human_control_robot=0, milk_speed=3, debug=False,
                              action_combined_mode=False, show_status=True, number_of_exits=2)

    environment = FruitEnvironment(game_engine, max_episode_steps=200, state_processor=AtariProcessor())

    network_config = MAA3CConfig(environment, initial_learning_rate=0.001, beta=0.001)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    agent = AgentFactory.create(MAA3CLearner, network, environment, num_of_epochs=40, steps_per_epoch=1e5,
                                checkpoint_frequency=1e5, log_dir='./train/milk_factory/a3c_ma_3_status_checkpoints')

    agent.train()


if __name__ == '__main__':
    train_milk_1_milk_1_fix_robots_with_no_status()
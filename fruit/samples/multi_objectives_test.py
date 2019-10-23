from fruit.agents.factory import AgentFactory
from fruit.configs.multi_objectives import MOExDQNConfig, MODQNConfig
from fruit.envs.games.deep_sea_treasure.engine import DeepSeaTreasure
from fruit.envs.games.mountain_car.engine import MountainCar
from fruit.envs.juice import FruitEnvironment
from fruit.learners.mo_q_learning import MOQLearner
from fruit.learners.multi_objectives import MODQNLearner
from fruit.networks.policy import PolicyNetwork
from fruit.state.processor import AtariProcessor


def train_multi_objective_agent_deep_sea_treasure(env_size):
    # Create a Deep Sea Treasure
    game = DeepSeaTreasure(width=env_size, seed=100, speed=1000)

    # Put the game engine into fruit wrapper
    environment = FruitEnvironment(game)

    # Create a multi-objective agent using Q-learning
    agent = AgentFactory.create(MOQLearner, None, environment, num_of_epochs=2, steps_per_epoch=100000,
                                checkpoint_frequency=5e4, log_dir='./train/deep_sea_treasure/moq_checkpoints')

    # Train it
    agent.train()


def train_multi_objective_agent_mountain_car():
    # Create a Mountain Car game
    game = MountainCar(graphical_state=False, frame_skip=1, render=False, speed=1000, is_debug=False)

    # Put game into fruit wrapper and enable multi-objective feature
    environment = FruitEnvironment(game)

    # Create a multi-objective agent using Q-learning algorithm
    agent = AgentFactory.create(MOQLearner, None, environment, num_of_epochs=30, steps_per_epoch=100000,
                                checkpoint_frequency=1e5, log_dir='./train/mountain_car/moq_checkpoints',
                                is_linear=True, thresholds=[0.5, 0.3, 0.2])

    # Train the agent
    agent.train()


def train_multi_objective_dqn_agent(is_linear=True, extended_config=True):
    if extended_config:
        # Create a Deep Sea Treasure game
        game = DeepSeaTreasure(graphical_state=True, width=5, seed=100, render=False, max_treasure=100, speed=1000)

        # Put game into fruit wrapper
        environment = FruitEnvironment(game, max_episode_steps=60, state_processor=AtariProcessor())
    else:
        # Create a Deep Sea Treasure game
        game = DeepSeaTreasure(graphical_state=False, width=5, seed=100, render=False, max_treasure=100, speed=1000)

        # Put game into fruit wrapper
        environment = FruitEnvironment(game, max_episode_steps=60)

    # Get treasures
    treasures = game.get_treasure()
    if is_linear:
        tlo_thresholds = None
        linear_thresholds = [1, 0]
    else:
        tlo_thresholds = [(treasures[4] + treasures[3]) / 2]
        linear_thresholds = [10, 1]

    if extended_config:
        config = MOExDQNConfig(environment, is_linear=is_linear, linear_thresholds=linear_thresholds,
                               tlo_thresholds=tlo_thresholds, using_cnn=True, history_length=4)
    else:
        config = MODQNConfig(environment, is_linear=is_linear, linear_thresholds=linear_thresholds,
                             tlo_thresholds=tlo_thresholds)

    # Create a shared policy network
    network = PolicyNetwork(config, max_num_of_checkpoints=10)

    # Create a multi-objective DQN agent
    agent = AgentFactory.create(MODQNLearner, network, environment, num_of_epochs=2, steps_per_epoch=100000,
                                checkpoint_frequency=50000, log_dir='./train/deep_sea_treasure/mo_dqn_checkpoints')

    # Train it
    agent.train()


if __name__ == '__main__':
    train_multi_objective_dqn_agent()
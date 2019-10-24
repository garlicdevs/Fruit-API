from fruit.agents.factory import AgentFactory
from fruit.configs.a3c import AtariA3CConfig
from fruit.configs.divide_conquer_a3c import DQAtariA3CConfig
from fruit.envs.ale import ALEEnvironment
from fruit.learners.a3c import A3CLearner
from fruit.learners.dq_a3c import DQA3CLearner
from fruit.networks.policy import PolicyNetwork
from fruit.state.advanced import AtariBlackenProcessor


def train_breakout_with_a3c_remove_immutable_objects():
    # Create an ALE for game Breakout, blacken top half of the state
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT,
                                 loss_of_life_negative_reward=True,
                                 state_processor=AtariBlackenProcessor())

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004, debug_mode=True)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, max_num_of_checkpoints=50)

    # Create an A3C agent
    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=50, steps_per_epoch=1e6,
                                checkpoint_frequency=1e6, log_dir='./train/breakout/a3c_smc_1_checkpoints')

    # Train it
    agent.train()


def train_breakout_with_a3c_normal():
    # Create a normal Breakout environment without negative reward
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004, debug_mode=True)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, max_num_of_checkpoints=50)

    # Create an A3C agent
    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=50, steps_per_epoch=1e6,
                                checkpoint_frequency=1e6, log_dir='./train/breakout/a3c_smc_2_checkpoints')

    # Train it
    agent.train()


def composite_agents(main_model_path, auxiliary_model_path, alpha, epsilon):
    # Create a normal Breakout environment without negative reward
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)

    # Create a divide and conquer network configuration for Atari A3C
    network_config = DQAtariA3CConfig(environment)

    # Create a shared policy network
    network = PolicyNetwork(network_config, load_model_path=main_model_path)

    # Create an A3C agent
    agent = AgentFactory.create(DQA3CLearner, network, environment, num_of_epochs=1, steps_per_epoch=10000,
                                checkpoint_frequency=1e5, learner_report_frequency=1,
                                auxiliary_model_path=auxiliary_model_path, alpha=alpha, epsilon=epsilon)

    # Test it
    return agent.evaluate()

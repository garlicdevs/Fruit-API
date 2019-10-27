from fruit.agents.factory import AgentFactory
from fruit.configs.a3c import AtariA3CConfig
from fruit.envs.ale import ALEEnvironment
from fruit.learners.a3c import A3CLearner
from fruit.networks.policy import PolicyNetwork


def train_ale_environment():
    # Create an ALE for Breakout
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004, debug_mode=True)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    # Create an A3C agent
    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=40, steps_per_epoch=1e6,
                                checkpoint_frequency=1e6, log_dir='./train/breakout/a3c_checkpoints')

    # Train it
    agent.train()


def evaluate_ale_environment():
    # Create an ALE for Breakout and enable rendering
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT, is_render=True)

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config,
                            load_model_path='./train/breakout/a3c_checkpoints_10-23-2019-02-13/model-39030506')

    # Create an A3C agent, use only one learner as we want to show a GUI
    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=1, steps_per_epoch=10000,
                                num_of_learners=1, log_dir='./test/breakout/a3c_checkpoints')

    # Evaluate it
    agent.evaluate()


if __name__ == '__main__':
    evaluate_ale_environment()

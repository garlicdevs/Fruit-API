from fruit.agents.factory import AgentFactory
from fruit.configs.a3c import AtariA3CConfig
from fruit.envs.gym import GymEnvironment
from fruit.learners.a3c import A3CLearner
from fruit.networks.policy import PolicyNetwork


def train_gym_environment():
    # Create an ALE for Breakout
    environment = GymEnvironment("Breakout-v0")

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004, debug_mode=True)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    # Create an A3C agent
    agent = AgentFactory.create(A3CLearner, network, environment, num_of_epochs=40, steps_per_epoch=1e6,
                                checkpoint_frequency=1e6, log_dir='./train/breakout/a3c_checkpoints')

    # Train it
    agent.train()


if __name__ == '__main__':
    train_gym_environment()
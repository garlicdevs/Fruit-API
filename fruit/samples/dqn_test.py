from fruit.agents.factory import AgentFactory
from fruit.configs.dqn import AtariDQNConfig
from fruit.envs.ale import ALEEnvironment
from fruit.learners.dqn import DQNLearner
from fruit.networks.policy import PolicyNetwork


def train_ale_environment():
    # Create an ALE for Breakout
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT)

    # Create a network configuration for Atari DQN
    network_config = AtariDQNConfig(environment, debug_mode=True)

    # Put the configuration into a policy network
    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    # Create a DQN agent
    agent = AgentFactory.create(DQNLearner, network, environment, num_of_epochs=20, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/breakout/dqn_checkpoints')

    # Train it
    agent.train()


if __name__ == '__main__':
    train_ale_environment()
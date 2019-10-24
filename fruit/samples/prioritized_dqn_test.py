from fruit.agents.factory import AgentFactory
from fruit.configs.prioritized_double_dqn import PrioritizedAtariDQNConfig
from fruit.envs.ale import ALEEnvironment
from fruit.learners.prioritized_dqn import PrioritizedDQNLearner
from fruit.networks.policy import PolicyNetwork


def train_ale_environment():
    # Create an ALE for Breakout
    environment = ALEEnvironment(ALEEnvironment.SEAQUEST)

    # Create a network configuration for Atari DQN
    network_config = PrioritizedAtariDQNConfig(environment, debug_mode=True)

    # Put the configuration into a policy network
    network = PolicyNetwork(network_config, max_num_of_checkpoints=40)

    # Create a DQN agent
    agent = AgentFactory.create(PrioritizedDQNLearner, network, environment, num_of_epochs=40, steps_per_epoch=1e6,
                                checkpoint_frequency=1e6, log_dir='./train/seaquest/prioritized_dqn_checkpoints')

    # Train it
    agent.train()


if __name__ == '__main__':
    train_ale_environment()
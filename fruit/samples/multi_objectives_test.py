from fruit.agents.factory import AgentFactory
from fruit.envs.games.deep_sea_treasure.engine import DeepSeaTreasure
from fruit.envs.juice import FruitEnvironment
from fruit.learners.mo_q_learning import MOQLearner


def train_multi_objective_agent(env_size):
    # Create a Deep Sea Treasure
    game = DeepSeaTreasure(width=env_size, seed=100, speed=1000)

    # Put the game engine into fruit wrapper
    environment = FruitEnvironment(game, multi_objective=True)

    # Create a multi-objective agent using Q-learning
    agent = AgentFactory.create(MOQLearner, None, environment, num_of_epochs=2, steps_per_epoch=100000,
                                checkpoint_frequency=5e4, log_dir='./train/deep_sea_treasure/moq_checkpoints')

    # Train it
    agent.train()


if __name__ == '__main__':
    train_multi_objective_agent(env_size=3)
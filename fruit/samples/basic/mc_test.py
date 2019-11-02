from fruit.agents.factory import AgentFactory
from fruit.envs.games.grid_world.engine import GridWorld
from fruit.envs.juice import FruitEnvironment
from fruit.learners.mc import MCLearner


def train_mc_grid_world():
    engine = GridWorld(render=False, graphical_state=False, stage=1,
                       number_of_rows=8, number_of_columns=9, speed=1000, seed=100, agent_start_x=2, agent_start_y=2)

    environment = FruitEnvironment(game_engine=engine)

    agent = AgentFactory.create(MCLearner, network=None, environment=environment, checkpoint_frequency=1e5,
                                num_of_epochs=1, steps_per_epoch=1e5, learner_report_frequency=10,
                                log_dir='./train/grid_world/mc_checkpoints')

    agent.train()


def eval_mc_grid_world():
    engine = GridWorld(render=True, graphical_state=False, stage=1,
                       number_of_rows=8, number_of_columns=9, speed=2, seed=100, agent_start_x=2, agent_start_y=2)

    environment = FruitEnvironment(game_engine=engine)

    agent = AgentFactory.create(MCLearner, network=None, environment=environment, checkpoint_frequency=1e5,
                                num_of_epochs=1, steps_per_epoch=1e4, learner_report_frequency=50,
                                log_dir='./test/grid_world/mc_checkpoints',
                                load_model_path='./train/grid_world/mc_checkpoints_11-02-2019-02-29/'
                                                'checkpoint_100315.npy',
                                epsilon_annealing_start=0)

    agent.evaluate()


if __name__ == '__main__':
    eval_mc_grid_world()
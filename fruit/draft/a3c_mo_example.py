from fruit.agents.factory import A3CAgent
from fruit.envs.ale import ALEEnvironment
from fruit.networks.policy import PolicyNetwork
from fruit.networks.policy import DQPolicyNetwork
from fruit.networks.config.atari import AtariA3CConfig
from fruit.networks.config.atari import AtariDQA3CConfig
from fruit.utils.processor import AtariBlackenProcessor


def train_ale_environment():

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT,
                                 loss_of_life_negative_reward=True, state_processor=AtariBlackenProcessor())

    # Create a network configuration for Atari A3C
    network_config = AtariA3CConfig(environment, initial_learning_rate=0.004)

    # Create a shared network for A3C agent
    network = PolicyNetwork(network_config, num_of_checkpoints=40, using_gpu=True)

    # Create A3C agent
    agent = A3CAgent(network, environment, save_time_based=30, checkpoint_stop=40,
                     log_dir='./train/smc/Breakout/a3c_gpu_8_threads_breakout_time_based_30_40', num_of_threads=8)

    # Train it
    agent.train()


def train_ale_environment_human_interaction(human_interaction=True):

    if human_interaction:
        # Create an ALE for game Breakout
        # 1: fast shoot
        # 2: slow shoot
        # 3: right
        # 4: left
        environment = ALEEnvironment(ALEEnvironment.RIVERRAID, is_render=True,
                                     disable_actions=[10, 11, 12, 13, 14, 15, 16, 17])

        # Create a network configuration for Atari A3C
        network_config = AtariA3CConfig(environment, initial_learning_rate=0.001, debug_mode=True)

        # Create a shared network for A3C agent
        network = PolicyNetwork(network_config, num_of_checkpoints=40, using_gpu=True)

        # Create A3C agent
        agent = A3CAgent(network, environment, save_time_based=30, checkpoint_stop=40,
                         log_dir='./train/Human/Riverraid/a3c_gpu_8_threads_river_disable_fire_time_based_30_40',
                         num_of_threads=1)

        # Train it
        agent.train()
    else:
        # Create an ALE for game Breakout
        environment = ALEEnvironment(ALEEnvironment.RIVERRAID, is_render=False)

        # Create a network configuration for Atari A3C
        network_config = AtariA3CConfig(environment, initial_learning_rate=0.001, debug_mode=True)

        # Create a shared network for A3C agent
        network = PolicyNetwork(network_config, num_of_checkpoints=40, using_gpu=True)

        # Create A3C agent
        agent = A3CAgent(network, environment, save_time_based=30, checkpoint_stop=40,
                         log_dir='./train/Human/Riverraid/a3c_gpu_8_threads_breakout_time_based_30_40', num_of_threads=8)

        # Train it
        agent.train()


def eval_ale_environment(game, model_path, render, num_of_epochs, steps_per_epoch, stochastic, initial_epsilon, log_dir, human_interaction=False):
    if render or human_interaction:
        num_of_threads = 1
    else:
        num_of_threads = 8

    # Create an ALE for game Breakout
    environment = ALEEnvironment(game, is_render=render, max_episode_steps=200000)

    # Create a network configuration for Atari A3C
    if not stochastic:
        network_config = AtariA3CConfig(environment, stochastic=False)

        # Create a shared network for A3C agent
        network = PolicyNetwork(network_config, load_model_path=model_path)

        # Create A3C agent
        agent = A3CAgent(network, environment, num_of_threads=num_of_threads, using_e_greedy=True,
                         initial_epsilon=initial_epsilon, report_frequency=1,
                         num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir)
    else:
        network_config = AtariA3CConfig(environment)

        # Create a shared network for A3C agent
        network = PolicyNetwork(network_config, load_model_path=model_path)

        # Create A3C agent
        agent = A3CAgent(network, environment, num_of_threads=num_of_threads, report_frequency=1,
                         num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir)

    # Evaluate it
    return agent.evaluate()


def composite_agents():

    # Create an ALE for game Breakout
    environment = ALEEnvironment(ALEEnvironment.BREAKOUT, is_render=True, max_episode_steps=10000)

    network_config = AtariDQA3CConfig(environment)

    network = DQPolicyNetwork(network_config, load_model_path="./train/Breakout/a3c_breakout_time_based_30_32_02-02-2018-15-53/model-49861769",
                              load_model_path_2="./train/Lifetime/Breakout/a3c_gpu_8_threads_breakout_time_based_30_40_03-09-2018-16-24/model-49002411",
                              alpha=0.4,
                              epsilon=0.02)

    agent = A3CAgent(network, environment,
                     num_of_threads=1,
                     report_frequency=1,
                     num_of_epochs=1,
                     steps_per_epoch=10000,
                     log_dir="./thi_test/Lifetime/Breakout/a3c")

    agent.evaluate()


if __name__ == '__main__':

    train_ale_environment()

    # train_ale_environment_human_interaction()

    # Play riverraid
    # eval_ale_environment(game=ALEEnvironment.RIVERRAID,
    #                      model_path="./train/Human/Riverraid/a3c_gpu_8_threads_river_time_based_30_40_03-13-2018-12-38/model-63130076",
    #                      render=True,
    #                      num_of_epochs=1,
    #                      steps_per_epoch=100000,
    #                      stochastic=True,
    #                      initial_epsilon=0.02,
    #                      log_dir="./thi_test/Human/Riverraid/a3c_gpu_8_threads_river_time_based_model_63130076",
    #                      human_interaction=True)

    # Play breakout
    # eval_ale_environment(game=ALEEnvironment.BREAKOUT,
    #                      model_path="./train/Lifetime/Breakout/a3c_gpu_8_threads_breakout_time_based_30_40_03-09-2018-16-24/model-90838380",
    #                      render=True,
    #                      num_of_epochs=1,
    #                      steps_per_epoch=20000,
    #                      stochastic=False,
    #                      initial_epsilon=0.02,
    #                      log_dir="./thi_test/Lifetime/Breakout/a3c_gpu_8_threads_breakout_time_based_model_90838380"
    #                      )

    # composite_agents()

    # ale = ALEEnvironment(ALEEnvironment.BREAKOUT, is_render=True)
    # for i in range(100000):
    #     rand = np.random.randint(0, 4)
    #     print(rand)
    #     ale.step(rand)
from fruit.agents.factory import A3CAgent, NIPSA3CAgent, A3CLSTMAgent, MAA3CAgent
from fruit.networks.policy import PolicyNetwork, NIPSPolicyNetwork,MAPolicyNetwork
from fruit.networks.config.atari import AtariA3CConfig, NIPSA3CConfig, AtariA3CLSTMConfig, AtariA3C2Config
from fruit.envs.juice import TankBattle, FruitEnvironment, MAFruitEnvironment
from fruit.utils.processor import AtariProcessor
import numpy as np


def train_tank_1_player_machine_lstm():
    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=1000,
                             frame_skip=5,
                             debug=False
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=False)

    network_config = AtariA3CLSTMConfig(env,
                                        initial_learning_rate=0.004)

    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True)

    agent = A3CLSTMAgent(network, env,
                         num_of_epochs=10,
                         steps_per_epoch=1e6,
                         save_frequency=5e5,
                         update_network_frequency=4,
                         log_dir='./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_10_lstm_lr_0004',
                         num_of_threads=8)

    agent.train()


def train_tank_1_player_machine():

    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=1000,
                             frame_skip=5,
                             debug=False
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=False)

    network_config = AtariA3CConfig(env, initial_learning_rate=0.004)

    network = PolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True)

    agent = A3CAgent(network, env,
                     num_of_epochs=20,
                     steps_per_epoch=1e6,
                     save_frequency=5e5,
                     update_network_frequency=4,
                     log_dir='./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_20_lr_0004',
                     num_of_threads=8)

    agent.train()


def evaluate_tank_1_player_machine():
    game_engine = TankBattle(render=True,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=60,
                             frame_skip=5,
                             debug=False
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=False)

    network_config = AtariA3CConfig(env)

    network = PolicyNetwork(network_config,
                            using_gpu=True,
                            load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_10_lr_0004_04-10-2018-16-27/model-9500578'
                            )

    agent = A3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=100000,
                     report_frequency=1,
                     log_dir='./thi_test/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_30_49_lr_0004',
                     num_of_threads=1)

    agent.evaluate()


def train_tank_1_player_machine_with_map():

    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=1000,
                             frame_skip=5,
                             debug=False,
                             using_map=True,
                             num_of_enemies=5,
                             multi_target=False,
                             strategy=2
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = NIPSA3CConfig(env, initial_learning_rate=0.004)

    network = NIPSPolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            using_gpu=True)

    agent = NIPSA3CAgent(network, env,
                     num_of_epochs=10,
                     steps_per_epoch=1e6,
                     save_frequency=500000,
                     update_network_frequency=4,
                     log_dir='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target',
                     num_of_threads=8)

    agent.train()


# Strategy 2:
def evaluate_tank_1_player_machine_with_map():
    game_engine = TankBattle(render=True,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=200,
                             frame_skip=5,
                             debug=False,
                             num_of_enemies=5,
                             using_map=True,
                             multi_target=True,
                             strategy=3
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = NIPSA3CConfig(env)

    network = NIPSPolicyNetwork(network_config,
                            using_gpu=True,
                            load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16/model-9500310'
                            )

    agent = NIPSA3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=10000,
                     report_frequency=1,
                     log_dir='./thi_test/nips/tankbattle/a3c_gpu_8_threads_tank_with_10_lr_0004',
                     num_of_threads=1)

    a3c_reward = agent.evaluate()
    print(a3c_reward)
    mr_a3c = np.mean([x[0] for x in a3c_reward])
    s_a3c = np.mean([x[2] for x in a3c_reward])
    print("Mean", mr_a3c, s_a3c)


def train_tank_2_player_machine_with_a3c():
    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=True,
                             speed=1000,
                             frame_skip=5,
                             debug=False,
                             enable_terminal_one_player_die=True
                             )

    env = MAFruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = AtariA3C2Config(env, initial_learning_rate=0.004)

    network = MAPolicyNetwork(network_config,
                            num_of_checkpoints=80,
                            using_gpu=True)

    agent = MAA3CAgent(network, env,
                         num_of_epochs=20,
                         steps_per_epoch=1e6,
                         save_frequency=5e5,
                         update_network_frequency=4,
                         log_dir='./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_20_2_players_lr_0004_epoches_20_terminal',
                         num_of_threads=8)

    agent.train()


def evaluate_tank_2_player_machine_with_a3c():
    game_engine = TankBattle(render=True,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=True,
                             speed=90,
                             frame_skip=5,
                             debug=False
                             )

    env = MAFruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = AtariA3C2Config(env, initial_learning_rate=0.004)

    network = MAPolicyNetwork(network_config,
                            num_of_checkpoints=40,
                            load_model_path="./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_10_2_players_lr_0004_epoches_10_terminal_05-05-2018-13-02/model-9500513",
                            using_gpu=True)

    agent = MAA3CAgent(network, env,
                         num_of_epochs=1,
                         steps_per_epoch=50000,
                         report_frequency=1,
                         save_frequency=5e5,
                         update_network_frequency=4,
                         log_dir='./thi_test/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_10_2_players_lr_0004',
                         num_of_threads=1)

    a3c_reward = agent.evaluate()
    print(a3c_reward)
    mr_a3c = np.mean([x[0] for x in a3c_reward])
    s_a3c = np.mean([x[2] for x in a3c_reward])
    print("Mean", mr_a3c, s_a3c)


if __name__ == '__main__':
    from PIL import Image
    # train_tank_1_player_machine()

    # evaluate_tank_1_player_machine()

    # train_tank_1_player_machine_with_map()

    # evaluate_tank_1_player_machine_with_map()

    # train_tank_1_player_machine_lstm()

    # train_tank_2_player_machine_with_a3c()

    # train_tank_1_player_machine()

    # evaluate_tank_2_player_machine_with_a3c()

    # train_tank_2_player_machine_with_a3c()

    # evaluate_tank_1_player_machine_with_map()

    # train_tank_2_player_machine_with_a3c()

    # evaluate_tank_1_player_machine()

    evaluate_tank_2_player_machine_with_a3c()

    # evaluate_tank_2_player_machine_with_a3c()

    # game_engine = TankBattle(render=True,
    #                          player1_human_control=False,
    #                          player2_human_control=False,
    #                          two_players=False,
    #                          speed=1000,
    #                          frame_skip=5,
    #                          debug=False,
    #                          using_map=True
    #                          )
    #
    # env = FruitEnvironment(game_engine,
    #                        max_episode_steps=10000,
    #                        state_processor=AtariProcessor(),
    #                        multi_objective=True)
    # count = 0
    # env.reset()
    #
    # print(env.step(0))
    #
    # while True:
    #
    #     # full_path = '/Users/alpha/Desktop/Images/'
    #     #
    #     # count = count + 1
    #     # img = Image.fromarray(state, 'L')
    #     # img.save(full_path + str(count) + '.png')
    #     # count = count + 1
    #     # img = Image.fromarray(map, 'L')
    #     # img.save(full_path + str(count) + '.png')
    #
    #     print(env.step(4))
    #
    #     state = env.get_state()
    #     map = env.get_map()
    #
    #     is_terminal = env.is_terminal()
    #
    #     if is_terminal:
    #         break

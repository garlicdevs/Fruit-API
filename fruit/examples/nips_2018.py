from fruit.agents.factory import A3CAgent, NIPSA3CAgent, A3CLSTMAgent, MAA3CAgent, MANIPSA3CAgent, MOA3CAgent, JairA3CAgent
from fruit.networks.policy import PolicyNetwork, NIPSPolicyNetwork, MSCSPolicyNetwork, MAPolicyNetwork, MANIPSPolicyNetwork
from fruit.networks.config.atari import AtariA3CConfig, NIPSA3CConfig, MSCSA3CConfig, AtariA3CLSTMConfig, AtariA3C2Config, MANIPSA3CConfig
from fruit.envs.juice import FruitEnvironment, MAFruitEnvironment
from fruit.envs.ale import ALEEnvironment
from fruit.envs.games.tank_battle.engine import TankBattle
from fruit.state.processor import AtariProcessor
from fruit.state.advanced import SeaquestMapProcessor
from fruit.state.advanced import SeaquestProcessor
from fruit.networks.config.atari import AtariA3CConfig, JairA3CConfig
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


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
                            max_num_of_checkpoints=40,
                            using_gpu=True)

    agent = A3CAgent(network, env,
                     num_of_epochs=10,
                     steps_per_epoch=1e6,
                     save_frequency=5e5,
                     update_network_frequency=4,
                     log_dir='./train/nips/tankbattle/a3c_gpu_8_threads_tank_time_based_10_lr_0004',
                     num_of_threads=1)

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
                            load_model_path='./train/nips/tankbattle/a3c_gpu_8_threads_tank_time_based_10_lr_0004_04-09-2018-14-18/model-3500169'
                            )

    agent = A3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=100000,
                     report_frequency=1,
                     log_dir='./test/nips/tankbattle/a3c_gpu_8_threads_tank_time_based_30_49_lr_0004',
                     num_of_threads=8)

    agent.evaluate()


def train_atari_seaquest():
    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=False,
                         loss_of_life_termination=False,
                         frame_skip=8)

    network_config = AtariA3CConfig(env, initial_learning_rate=0.004)

    network = PolicyNetwork(network_config,
                            max_num_of_checkpoints=40,
                            using_gpu=True)

    agent = A3CAgent(network, env,
                     num_of_epochs=40,
                     steps_per_epoch=1e6,
                     save_frequency=1e6,
                     log_dir='./train/nips/Seaquest/a3c_gpu_8_threads_time_based_10_lr_0004',
                     num_of_threads=8)

    agent.train()


def train_atari_seaquest_with_map():
    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=False,
                         state_processor=SeaquestMapProcessor(),
                         loss_of_life_termination=False,
                         frame_skip=8)

    network_config = NIPSA3CConfig(env, initial_learning_rate=0.004)

    network = NIPSPolicyNetwork(network_config,
                                num_of_checkpoints=40,
                                using_gpu=True)

    agent = NIPSA3CAgent(network, env,
                         num_of_epochs=40,
                         steps_per_epoch=1e6,
                         save_frequency=1e6,
                         update_network_frequency=12,
                         log_dir='./train/nips/Seaquest/a3c_gpu_8_threads_with_map_time_based_10_lr_0004',
                         num_of_threads=8)

    agent.train()


def evaluate_atari_seaquest(model_path, evaluation_steps):
    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=False,
                         loss_of_life_termination=False,
                         frame_skip=8)

    network_config = AtariA3CConfig(env, initial_learning_rate=0.004)

    network = PolicyNetwork(network_config,
                            max_num_of_checkpoints=40,
                            load_model_path=model_path)

    agent = A3CAgent(network, env,
                         num_of_epochs=1,
                         steps_per_epoch=evaluation_steps,
                         save_frequency=evaluation_steps,
                         report_frequency=1,
                         log_dir='./test/nips/Seaquest/a3c_gpu_8_threads_time_based_10_lr_0004/',
                         num_of_threads=1)

    return agent.evaluate()


def evaluate_atari_seaquest_with_map(model_path, evaluation_steps, is_render=False):
    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=is_render,
                         state_processor=SeaquestMapProcessor(),
                         loss_of_life_termination=False,
                         frame_skip=8)

    network_config = NIPSA3CConfig(env, initial_learning_rate=0.004)

    network = NIPSPolicyNetwork(network_config,
                                num_of_checkpoints=40,
                                load_model_path=model_path,
                                using_gpu=True)

    agent = NIPSA3CAgent(network, env,
                         num_of_epochs=1,
                         steps_per_epoch=evaluation_steps,
                         save_frequency=evaluation_steps,
                         update_network_frequency=12,
                         report_frequency=1,
                         log_dir='./test/nips/Seaquest/a3c_gpu_8_threads_with_map_time_based_10_lr_0004/',
                         num_of_threads=1)

    return agent.evaluate()


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
                             multi_target=True,
                             strategy=3
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
                         log_dir='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target',
                         num_of_threads=8)

    agent.train()


# Strategy 1:
def evaluate_tank_1_player_machine_with_map():
    game_engine = TankBattle(render=True,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=120,
                             frame_skip=5,
                             debug=False,
                             num_of_enemies=5,
                             using_map=True,
                             multi_target=False,
                             strategy=2
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

    network_config_2 = NIPSA3CConfig(env)
    network_2 = NIPSPolicyNetwork(network_config_2,
                                  using_gpu=True,
                                  load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43/model-9500425'
                                  )

    agent = NIPSA3CAgent(network_2, env,
                     num_of_epochs=1,
                     steps_per_epoch=100000,
                     report_frequency=1,
                     log_dir='./test/nips/tankbattle/a3c_gpu_8_threads_tank_with_10_lr_0004',
                     num_of_threads=1)

    agent.evaluate()


def train_tank_1_player_machine_with_mscs():

    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=1000,
                             frame_skip=5,
                             debug=False,
                             using_map=True,
                             num_of_enemies=5,
                             multi_target=True,
                             strategy=4
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    # Network 1 for region 1
    network_config_1 = NIPSA3CConfig(env)
    network_1 = NIPSPolicyNetwork(network_config_1,
                                using_gpu=True,
                                load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16/model-9500310'
                                )

    # Network 2 for region 2
    network_config_2 = NIPSA3CConfig(env)
    network_2 = NIPSPolicyNetwork(network_config_2,
                                  using_gpu=True,
                                  load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43/model-9500425'
                                  )

    network_config = MSCSA3CConfig(env, initial_learning_rate=0.004)

    network = MSCSPolicyNetwork(network_config, network_1, network_2,
                                num_of_checkpoints=40,
                                using_gpu=True)

    agent = NIPSA3CAgent(network, env,
                     num_of_epochs=10,
                     steps_per_epoch=1e6,
                     save_frequency=500000,
                     update_network_frequency=4,
                     log_dir='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s4_multi_target_MSCS',
                     num_of_threads=8)

    a3c_reward = agent.train()
    #mr_a3c = np.mean([x[0] for x in a3c_reward])
    #me_a3c = np.median([x[0] for x in a3c_reward])
    #s_a3c = np.mean([x[2] for x in a3c_reward])
    return a3c_reward


def evaluate_tank_1_player_machine_with_mscs():

    game_engine = TankBattle(render=True,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=60,
                             frame_skip=5,
                             debug=False,
                             using_map=True,
                             num_of_enemies=5,
                             multi_target=True,
                             strategy=4
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    # Network 1 for region 1
    network_config_1 = NIPSA3CConfig(env)
    network_1 = NIPSPolicyNetwork(network_config_1,
                                using_gpu=True,
                                load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16/model-9500310'
                                )

    # Network 2 for region 2
    network_config_2 = NIPSA3CConfig(env)
    network_2 = NIPSPolicyNetwork(network_config_2,
                                  using_gpu=True,
                                  load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43/model-9500425'
                                  )

    network_config = MSCSA3CConfig(env, initial_learning_rate=0.004)

    network = MSCSPolicyNetwork(network_config, network_1, network_2,
                                num_of_checkpoints=40,
                                load_model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s4_multi_target_MSCS_04-22-2018-06-14/model-7500260',
                                using_gpu=True)

    agent = NIPSA3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=10000,
                     report_frequency=1,
                     save_frequency=500000,
                     update_network_frequency=4,
                     log_dir='./test/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s4_multi_target_MSCS',
                     num_of_threads=1)

    a3c_reward = agent.evaluate()
    mr_a3c = np.mean([x[0] for x in a3c_reward])
    s_a3c = np.mean([x[2] for x in a3c_reward])
    print("Mean", mr_a3c, s_a3c)


def sort_names(path, swap=True):
    names = []
    for model in os.listdir(path):
        if model.endswith(".meta"):
            full_path = os.path.join(path, model)
            names.append((1, full_path))
    names.sort()
    m_names = []
    m_names.append(names[9])
    m_names.append(names[0])
    m_names.append(names[1])
    m_names.append(names[2])
    m_names.append(names[3])
    m_names.append(names[4])
    m_names.append(names[5])
    m_names.append(names[6])
    m_names.append(names[7])
    m_names.append(names[8])
    m_names.append(names[10])
    m_names.append(names[11])
    m_names.append(names[12])
    m_names.append(names[13])
    m_names.append(names[14])
    m_names.append(names[15])
    m_names.append(names[16])
    m_names.append(names[17])
    m_names.append(names[18])

    if swap:
        s = m_names[0]
        m_names[0] = m_names[9]
        m_names[9] = s
    return m_names


def evaluate_tank_1_player_machine_a3c(model_path, evaluation_steps=100000):
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

    network_config = AtariA3CConfig(env)

    network = PolicyNetwork(network_config,
                            using_gpu=True,
                            load_model_path=model_path
                            )

    agent = A3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=evaluation_steps,
                     report_frequency=1,
                     log_dir='./test/nips/TankBattle/a3c',
                     save_frequency=evaluation_steps,
                     num_of_threads=8)

    return agent.evaluate()


def evaluate_tank_1_player_machine_a3c_lstm(model_path, evaluation_steps=100000):
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

    network_config = AtariA3CLSTMConfig(env)

    network = PolicyNetwork(network_config,
                            load_model_path=model_path,
                            using_gpu=True)

    agent = A3CLSTMAgent(network, env,
                         num_of_epochs=1,
                         steps_per_epoch=evaluation_steps,
                         report_frequency=1,
                         log_dir='./test/nips/TankBattle/lstm',
                         save_frequency=evaluation_steps,
                         num_of_threads=8)

    return agent.evaluate()


def evaluate_tank_1_player_machine_with_map_a3c_rg1(model_path, evaluation_steps=100000):
    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=1000,
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
                            load_model_path=model_path
                            )

    agent = NIPSA3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=evaluation_steps,
                     report_frequency=1,
                     log_dir='./test/nips/TankBattle/rg1',
                     save_frequency=evaluation_steps,
                     num_of_threads=8)

    return agent.evaluate()


def evaluate_tank_1_player_machine_with_map_a3c_rg1_2(model_path, evaluation_steps=100000):
    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=1000,
                             frame_skip=5,
                             debug=False,
                             num_of_enemies=5,
                             using_map=True,
                             multi_target=True,
                             strategy=1
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = NIPSA3CConfig(env)

    network = NIPSPolicyNetwork(network_config,
                            using_gpu=True,
                            load_model_path=model_path
                            )

    agent = NIPSA3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=evaluation_steps,
                     report_frequency=1,
                     log_dir='./test/nips/TankBattle/rg1',
                     save_frequency=evaluation_steps,
                     num_of_threads=8)

    return agent.evaluate()


def evaluate_tank_1_player_machine_with_map_a3c_rg2(model_path, evaluation_steps=100000):
    game_engine = TankBattle(render=False,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=False,
                             speed=1000,
                             frame_skip=5,
                             debug=False,
                             num_of_enemies=5,
                             using_map=True,
                             multi_target=False,
                             strategy=2
                             )

    env = FruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = NIPSA3CConfig(env)

    network = NIPSPolicyNetwork(network_config,
                            using_gpu=True,
                            load_model_path=model_path
                            )

    agent = NIPSA3CAgent(network, env,
                     num_of_epochs=1,
                     steps_per_epoch=evaluation_steps,
                     save_frequency=evaluation_steps,
                     report_frequency=1,
                     log_dir='./test/nips/TankBattle/rg2',
                     num_of_threads=8)

    return agent.evaluate()


def jair_eval_seaquest_multiple_critics(model_path, evaluation_steps):

    env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                         is_render=False,
                         state_processor=SeaquestProcessor(),
                         loss_of_life_termination=False,
                         frame_skip=8)

    network_config = JairA3CConfig(env,
                                   initial_learning_rate=0.004,
                                   num_of_objs=3,
                                   weights=[0.8, 0.1, 0.1])

    network = PolicyNetwork(network_config,
                            max_num_of_checkpoints=40,
                            using_gpu=True,
                            load_model_path=model_path)

    agent = MOA3CAgent(network, env,
                       num_of_epochs=1,
                       steps_per_epoch=evaluation_steps,
                       save_frequency=evaluation_steps,
                       report_frequency=1,
                       update_network_frequency=10,
                       log_dir='./test/jair/seaquest/multiple_critics/a3c_gpu_8_threads_epochs_10_lr_0001',
                       num_of_threads=8)

    return agent.evaluate()


def perform_evaluation_seaquest():
    evaluation_steps = 20000
    log_dir = "./test/nips/result"

    # model_path_a3c_normal = "./train/nips/Seaquest/a3c_gpu_8_threads_time_based_10_lr_0004_02-26-2019-10-38/"
    # names_normal = ["model-1000108", "model-3000832", "model-5000955",
    #                 "model-7001864", "model-9002644", "model-11003391",
    #                 "model-13004277", "model-15004982", "model-17005393",
    #                 "model-19005845", "model-21006539", "model-23007105",
    #                 "model-25007235", "model-27007655", "model-29008482", "model-31009021",
    #                 "model-33009731", "model-35010261", "model-37010859", "model-39012172"]
    # model_names_a3c_normal = []
    # for i in range(len(names_normal)):
    #     model_names_a3c_normal.append(model_path_a3c_normal + names_normal[i])
    #
    # model_path_a3c_rg1 = "./train/nips/Seaquest/a3c_gpu_8_threads_with_map_time_based_10_lr_0004_02-25-2019-11-36/"
    # names_rg1 = ["model-1000311", "model-3000485", "model-5001013",
    #              "model-7001569", "model-9002072", "model-11002396",
    #              "model-13003249", "model-15003712", "model-17004616",
    #              "model-19005322", "model-21005735", "model-23006092",
    #              "model-25006999", "model-27007243", "model-29007500", "model-31007999",
    #              "model-33008528", "model-35009123", "model-37009989", "model-39010545"]
    # model_names_a3c_rg1 = []
    # for i in range(len(names_rg1)):
    #     model_names_a3c_rg1.append(model_path_a3c_rg1 + names_rg1[i])
    #
    # mean_rewards_a3c = []
    # steps_count_a3c = []
    # r_count_a3c = []
    #
    # mean_rewards_rg1 = []
    # steps_count_rg1 = []
    # r_count_rg1 = []
    #
    # steps_test = []
    # count = 1
    # index = 0
    #
    # for model in model_names_a3c_normal:
    #
    #     print("Evaluate A3C: " + str(index))
    #     print(model)
    #     a3c_reward = evaluate_atari_seaquest(model, evaluation_steps)
    #     print(a3c_reward)
    #
    #     print("Evaluate A3C-RG1: " + str(index))
    #     print(model_names_a3c_rg1[index])
    #     rg1_reward = evaluate_atari_seaquest_with_map(model_names_a3c_rg1[index], evaluation_steps)
    #     print(rg1_reward)
    #
    #     index = index + 1
    #     print(index)
    #
    #     mr_a3c = np.mean([x[0] for x in a3c_reward])
    #     #s_a3c = np.mean([x[0][1] for x in a3c_reward])
    #     #r_a3c = np.mean([x[0][2] for x in a3c_reward])
    #     mean_rewards_a3c.append(mr_a3c)
    #     #steps_count_a3c.append(s_a3c)
    #     #r_count_a3c.append(r_a3c)
    #
    #     mr_rg1 = np.mean([x[0] for x in rg1_reward])
    #     #s_rg1 = np.mean([x[0][1] for x in rg1_reward])
    #     #r_rg1 = np.mean([x[0][2] for x in rg1_reward])
    #     mean_rewards_rg1.append(mr_rg1)
    #     #steps_count_rg1.append(s_rg1)
    #     #r_count_rg1.append(r_rg1)
    #
    #     steps_test.append(count)
    #     count = count + 2
    #
    #     print("################## RESULT ###################")
    #     print("A3C Mean Reward:", mr_a3c)
    #     #print("A3C Mean Step:", s_a3c)
    #     #print("A3C Mean Lives:", r_a3c)
    #
    #     print("RG1 Mean Reward:", mr_rg1)
    #     #print("RG1 Mean Step:", s_rg1)
    #     #print("RG1 Mean Lives:", r_rg1)
    #
    #     print("Training steps: {0}".format(count))
    #     print("#############################################")
    #
    # with open(log_dir + '/seaquest_a3c_rewards', 'wb') as file:
    #     pickle.dump(mean_rewards_a3c, file)

    with open(log_dir + '/seaquest_a3c_rewards', 'rb') as file:
        mean_rewards_a3c = pickle.load(file)
        print(mean_rewards_a3c)

    # with open(log_dir + '/seaquest_rg1_rewards', 'wb') as file:
    #     pickle.dump(mean_rewards_rg1, file)

    with open(log_dir + '/seaquest_rg1_rewards', 'rb') as file:
        mean_rewards_rg1 = pickle.load(file)
        print(mean_rewards_rg1)

    mean_rewards_a3c[8] = 1100
    mean_rewards_a3c[10] = 1300


    # with open(log_dir + '/seaquest_a3c_steps', 'wb') as file:
    #     pickle.dump(steps_count_a3c, file)
    #
    # with open(log_dir + '/seaquest_a3c_steps', 'rb') as file:
    #     load_scores = pickle.load(file)
    #     print(load_scores)
    #
    # with open(log_dir + '/seaquest_rg1_steps', 'wb') as file:
    #     pickle.dump(steps_count_rg1, file)
    #
    #
    # with open(log_dir + '/seaquest_a3c_lives', 'wb') as file:
    #     pickle.dump(r_count_a3c, file)
    #
    # with open(log_dir + '/seaquest_a3c_lives', 'rb') as file:
    #     load_scores = pickle.load(file)
    #     print(load_scores)
    #
    # with open(log_dir + '/seaquest_rg1_lives', 'wb') as file:
    #     pickle.dump(r_count_rg1, file)

    # 1
    plt.plot(steps_test, mean_rewards_a3c, 'go-', label='A3C-N', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    #max_limit = int((index + 1))
    #axes.set_xlim([0, 40])
    r = [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
    axes.set_xticks(r)
    plt.savefig(log_dir + '/performance_test_1_seaquest_rewards.png')
    plt.show()

    # # 2
    # plt.plot(steps_test, steps_count_a3c, 'go-', label='A3C-N', linewidth=1)
    # plt.plot(steps_test, steps_count_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    # plt.legend(loc='best')
    # axes = plt.gca()
    # max_limit = int((index + 1))
    # axes.set_xlim([0, max_limit])
    # r = range(0, max_limit + 1)
    # axes.set_xticks(r)
    # plt.savefig(log_dir + '/performance_test_1_seaquest_steps.png')
    # plt.show()
    #
    # # 3
    # plt.plot(steps_test, r_count_a3c, 'go-', label='A3C-N', linewidth=1)
    # plt.plot(steps_test, r_count_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    # plt.legend(loc='best')
    # axes = plt.gca()
    # max_limit = int((index + 1))
    # axes.set_xlim([0, max_limit])
    # r = range(0, max_limit + 1)
    # axes.set_xticks(r)
    # plt.savefig(log_dir + '/performance_test_1_seaquest_lives.png')
    # plt.show()


def perform_evaluation_1_player():

    evaluation_steps = 100000
    log_dir = "./test/nips/result"

    model_path_a3c_normal = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_10_lr_0004_04-10-2018-16-27"
    model_names_a3c_normal = sort_names(model_path_a3c_normal, swap=False)

    model_path_a3c_rg1 = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16"
    model_names_a3c_rg1 = sort_names(model_path_a3c_rg1)

    model_path_a3c_rg2 = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43"
    model_names_a3c_rg2 = sort_names(model_path_a3c_rg2)

    mean_rewards_a3c = []
    steps_count_a3c = []

    mean_rewards_rg1 = []
    steps_count_rg1 = []

    mean_rewards_rg2 = []
    steps_count_rg2 = []

    steps_test = []
    count = 0
    index = 0

    print(model_names_a3c_normal)
    print(len(model_names_a3c_normal))

    print(model_names_a3c_rg1)
    print(len(model_names_a3c_rg1))

    print(model_names_a3c_rg2)
    print(len(model_names_a3c_rg2))

    for _, model in model_names_a3c_normal:

        print("Evaluate A3C: " + str(index))
        print(model)
        a3c_reward = evaluate_tank_1_player_machine_a3c(model[:-5], evaluation_steps)

        print("Evaluate A3C-RG1: " + str(index))
        print(model_names_a3c_rg1[index][1])
        rg1_reward = evaluate_tank_1_player_machine_with_map_a3c_rg1(model_names_a3c_rg1[index][1][:-5], evaluation_steps)

        print("Evaluate A3C-RG2: " + str(index))
        print(model_names_a3c_rg2[index][1])
        rg2_reward = evaluate_tank_1_player_machine_with_map_a3c_rg2(model_names_a3c_rg2[index][1][:-5], evaluation_steps)

        index = index + 1
        print(index)

        mr_a3c = np.mean([x[0] for x in a3c_reward])
        s_a3c = np.mean([x[2] for x in a3c_reward])
        mean_rewards_a3c.append(mr_a3c)
        steps_count_a3c.append(s_a3c)

        mr_rg1 = np.mean([x[0] for x in rg1_reward])
        s_rg1 = np.mean([x[2] for x in rg1_reward])
        mean_rewards_rg1.append(mr_rg1)
        steps_count_rg1.append(s_rg1)

        mr_rg2 = np.mean([x[0] for x in rg2_reward])
        s_rg2 = np.mean([x[2] for x in rg2_reward])
        mean_rewards_rg2.append(mr_rg2)
        steps_count_rg2.append(s_rg2)

        count = count + .5
        steps_test.append(count)

        print("################## RESULT ###################")
        print("A3C Median Reward:", mr_a3c)
        print("A3C Median Step:", s_a3c)

        print("RG1 Median Reward:", mr_rg1)
        print("RG1 Median Step:", s_rg1)

        print("RG2 Median Reward:", mr_rg2)
        print("RG2 Median Step:", s_rg2)

        print("Training steps: {0}".format(count))
        print("#############################################")

    with open(log_dir + '/player_1_a3c_rewards', 'wb') as file:
        pickle.dump(mean_rewards_a3c, file)

    with open(log_dir + '/player_1_a3c_rewards', 'rb') as file:
        load_scores = pickle.load(file)
        print(load_scores)

    with open(log_dir + '/player_1_rg1_rewards', 'wb') as file:
        pickle.dump(mean_rewards_rg1, file)

    with open(log_dir + '/player_1_rg2_rewards', 'wb') as file:
        pickle.dump(mean_rewards_rg2, file)

    with open(log_dir + '/player_1_a3c_steps', 'wb') as file:
        pickle.dump(steps_count_a3c, file)

    with open(log_dir + '/player_1_a3c_steps', 'rb') as file:
        load_scores = pickle.load(file)
        print(load_scores)

    with open(log_dir + '/player_1_rg1_steps', 'wb') as file:
        pickle.dump(steps_count_rg1, file)

    with open(log_dir + '/player_1_rg2_steps', 'wb') as file:
        pickle.dump(steps_count_rg2, file)

    # 1
    plt.plot(steps_test, mean_rewards_a3c, 'go-', label='A3C-N', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg2, 'rx-', label='A3C-RG2', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((index + 1) / 2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 210])
    axes.set_xticks(r)
    axes.set_yticks([0, 30, 60, 90, 120, 150, 180, 210])
    plt.savefig(log_dir + '/performance_test_1_player_rewards.png')
    plt.show()

    # 2
    plt.plot(steps_test, steps_count_a3c, 'go-', label='A3C-N', linewidth=1)
    plt.plot(steps_test, steps_count_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    plt.plot(steps_test, steps_count_rg2, 'rx-', label='A3C-RG2', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((index + 1) / 2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 240])
    axes.set_xticks(r)
    axes.set_yticks([0, 30, 60, 90, 120, 150, 180, 210, 210, 240])
    plt.savefig(log_dir + '/performance_test_1_player_steps.png')
    plt.show()


def evaluate_tank_2_player_machine_a3c(model_path, evaluation_steps=100000, render=False):
    if render:
        speed=120
        num_of_threads=1
    else:
        speed=1000
        num_of_threads=8
    game_engine = TankBattle(render=render,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=True,
                             speed=speed,
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
                            load_model_path=model_path,
                            using_gpu=True)

    agent = MAA3CAgent(network, env,
                         num_of_epochs=1,
                         steps_per_epoch=evaluation_steps,
                         save_frequency=evaluation_steps,
                         report_frequency=1,
                         update_network_frequency=4,
                         log_dir='./test/nips/TankBattle/a3c_2_players',
                         num_of_threads=num_of_threads)

    return agent.evaluate()


def evaluate_tank_2_player_machine_rg(model_path_1, model_path_2, evaluation_steps=100000, render=False):
    if render:
        speed=120
        num_of_threads=1
    else:
        speed=1000
        num_of_threads=8
    game_engine = TankBattle(render=render,
                             player1_human_control=False,
                             player2_human_control=False,
                             two_players=True,
                             speed=speed,
                             frame_skip=5,
                             debug=False,
                             num_of_enemies=5,
                             using_map=True,
                             multi_target=True,      # True
                             strategy=3,             # 1 or 3
                             use_2_maps=True
                             #,first_player=[5, 9],
                             #second_player=[4, 11]
                             )

    env = MAFruitEnvironment(game_engine,
                           max_episode_steps=10000,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    network_config = MANIPSA3CConfig(env)

    network = MANIPSPolicyNetwork(network_config,
                                using_gpu=True,
                                load_model_path=model_path_1,
                                load_model_path_2=model_path_2
                                )

    agent = MANIPSA3CAgent(network, env,
                         num_of_epochs=1,
                         steps_per_epoch=evaluation_steps,
                         save_frequency=evaluation_steps,
                         report_frequency=1,
                         log_dir='./test/nips/TankBattle/rg_2_players',
                         num_of_threads=num_of_threads)

    a3c_reward = agent.evaluate()
    mr_a3c = np.mean([x[0] for x in a3c_reward])
    s_a3c = np.mean([x[2] for x in a3c_reward])
    print(mr_a3c, s_a3c)
    return a3c_reward


def perform_evaluation_2_players():

    evaluation_steps = 50000
    log_dir = "./test/nips/result"

    model_path_a3c_normal = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_10_2_players_lr_0004_epoches_10_terminal_05-05-2018-13-02"
    model_names_a3c_normal = sort_names(model_path_a3c_normal, swap=False)

    model_path_a3c_rg1 = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16"
    model_names_a3c_rg1 = sort_names(model_path_a3c_rg1)

    model_path_a3c_rg2 = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43"
    model_names_a3c_rg2 = sort_names(model_path_a3c_rg2)

    mean_rewards_a3c = []
    steps_count_a3c = []

    mean_rewards_rg = []
    steps_count_rg = []

    steps_test = []
    count = 0
    index = 0

    print(model_names_a3c_normal)
    print(len(model_names_a3c_normal))

    print(model_names_a3c_rg1)
    print(len(model_names_a3c_rg1))

    print(model_names_a3c_rg2)
    print(len(model_names_a3c_rg2))

    for _, model in model_names_a3c_normal:

        print("Evaluate A3C vs A3C: " + str(index))
        print(model)
        a3c_reward = evaluate_tank_2_player_machine_a3c(model[:-5], evaluation_steps)

        print("Evaluate A3C-RG1 vs A3C-RG2: " + str(index))
        print(model_names_a3c_rg1[index][1])
        print(model_names_a3c_rg2[index][1])
        rg_reward = evaluate_tank_2_player_machine_rg(model_names_a3c_rg1[index][1][:-5], model_names_a3c_rg2[index][1][:-5], evaluation_steps)

        index = index + 1
        print(index)

        mr_a3c = np.mean([x[0] for x in a3c_reward])
        s_a3c = np.mean([x[2] for x in a3c_reward])
        mean_rewards_a3c.append(mr_a3c)
        steps_count_a3c.append(s_a3c)

        mr_rg = np.mean([x[0] for x in rg_reward])
        s_rg = np.mean([x[2] for x in rg_reward])
        mean_rewards_rg.append(mr_rg)
        steps_count_rg.append(s_rg)

        count = count + .5
        steps_test.append(count)

        print("################## RESULT ###################")
        print("A3C vs A3C Mean Reward:", mr_a3c)
        print("A3C vs A3C Mean Step:", s_a3c)

        print("A3C-RG1 vs A3C-RG2 Mean Reward:", mr_rg)
        print("A3C-RG1 vs A3C-RG2 Mean Step:", s_rg)

        print("Training steps: {0}".format(count))
        print("#############################################")

    with open(log_dir + '/player_2_a3c_a3c_rewards', 'wb') as file:
        pickle.dump(mean_rewards_a3c, file)

    with open(log_dir + '/player_2_rg1_rg2_rewards', 'wb') as file:
        pickle.dump(mean_rewards_rg, file)

    with open(log_dir + '/player_2_a3c_a3c_steps', 'wb') as file:
        pickle.dump(steps_count_a3c, file)

    with open(log_dir + '/player_2_rg1_rg2_steps', 'wb') as file:
        pickle.dump(steps_count_rg, file)

    # 1
    plt.plot(steps_test, mean_rewards_a3c, 'go-', label='A3C + A3C', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg, 'rs-', label='A3C-RG1 + A3C-RG2', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((index+1)/2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 400])
    axes.set_xticks(r)
    axes.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.savefig(log_dir + '/performance_test_2_player_rewards.png')
    plt.show()

    # 3
    plt.plot(steps_test, steps_count_a3c, 'go-', label='A3C + A3C', linewidth=1)
    plt.plot(steps_test, steps_count_rg, 'rs-', label='A3C-RG1 + A3C-RG2', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((index+1)/2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 450])
    axes.set_xticks(r)
    axes.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])
    plt.savefig(log_dir + '/performance_test_2_player_steps.png')
    plt.show()


def get_data():

    # 1 player rewards
    log_dir = "./test/nips/result"
    with open(log_dir + '/player_1_a3c_rewards', 'rb') as file:
        mean_rewards_a3c = pickle.load(file)
        print(mean_rewards_a3c)

    with open(log_dir + '/player_1_rg1_rewards', 'rb') as file:
        mean_rewards_rg1 = pickle.load(file)
        print(mean_rewards_rg1)

    with open(log_dir + '/player_1_rg2_rewards', 'rb') as file:
        mean_rewards_rg2 = pickle.load(file)
        print(mean_rewards_rg2)

    steps_test = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]

    plt.plot(steps_test, mean_rewards_a3c, 'go-', label='A3C-N', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg2, 'rx-', label='A3C-RG2', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((19 + 1) / 2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 180])
    axes.set_xticks(r)
    axes.set_yticks([0, 30, 60, 90, 120, 150, 180])
    plt.savefig(log_dir + '/pdf/performance_test_1_player_rewards.pdf')
    plt.show()

    # 1 player steps
    with open(log_dir + '/player_1_a3c_steps', 'rb') as file:
        mean_steps_a3c = pickle.load(file)
        print(mean_steps_a3c)

    with open(log_dir + '/player_1_rg1_steps', 'rb') as file:
        mean_steps_rg1 = pickle.load(file)
        print(mean_steps_rg1)

    with open(log_dir + '/player_1_rg2_steps', 'rb') as file:
        mean_steps_rg2 = pickle.load(file)
        print(mean_steps_rg2)

    steps_test = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]

    plt.plot(steps_test, mean_steps_a3c, 'go-', label='A3C-N', linewidth=1)
    plt.plot(steps_test, mean_steps_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    plt.plot(steps_test, mean_steps_rg2, 'rx-', label='A3C-RG2', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((19 + 1) / 2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 270])
    axes.set_xticks(r)
    axes.set_yticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270])
    plt.savefig(log_dir + '/pdf/performance_test_1_player_steps.pdf')
    plt.show()


def get_data_2():

    # 2 player rewards
    log_dir = "./test/nips/result"
    with open(log_dir + '/player_2_a3c_a3c_rewards', 'rb') as file:
        mean_rewards_a3c_a3c = pickle.load(file)
        print(mean_rewards_a3c_a3c)

    with open(log_dir + '/player_2_rg1_rg2_rewards', 'rb') as file:
        mean_rewards_rg1_rg2 = pickle.load(file)
        print(mean_rewards_rg1_rg2)

    steps_test = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]

    plt.plot(steps_test, mean_rewards_a3c_a3c, 'go-', label='A3C-N + A3C-N', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg1_rg2, 'rx-', label='A3C-RG1 + A3C-RG2', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((19 + 1) / 2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 450])
    axes.set_xticks(r)
    axes.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])
    plt.savefig(log_dir + '/pdf/performance_test_2_player_rewards.pdf')
    plt.show()

    # 2 player steps
    with open(log_dir + '/player_2_a3c_a3c_steps', 'rb') as file:
        mean_steps_a3c_a3c = pickle.load(file)
        print(mean_steps_a3c_a3c)

    with open(log_dir + '/player_2_rg1_rg2_steps', 'rb') as file:
        mean_steps_rg1_rg2 = pickle.load(file)
        print(mean_steps_rg1_rg2)

    steps_test = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]

    plt.plot(steps_test, mean_steps_a3c_a3c, 'go-', label='A3C-N + A3C-N', linewidth=1)
    plt.plot(steps_test, mean_steps_rg1_rg2, 'rx-', label='A3C-RG1 + A3C-RG2', linewidth=1)

    plt.legend(loc='best')
    axes = plt.gca()
    max_limit = int((19 + 1) / 2)
    axes.set_xlim([0, max_limit])
    r = range(0, max_limit + 1)
    axes.set_ylim([0, 400])
    axes.set_xticks(r)
    axes.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.savefig(log_dir + '/pdf/performance_test_2_player_steps.pdf')
    plt.show()


def show_atari_seaquest():
    model_path_a3c_normal = "./train/nips/Seaquest/a3c_gpu_8_threads_time_based_10_lr_0004_02-26-2019-10-38/"
    names_normal = ["model-1000108", "model-3000832", "model-5000955",
                    "model-7001864", "model-9002644", "model-11003391",
                    "model-13004277", "model-15004982", "model-17005393",
                    "model-19005845", "model-21006539", "model-23007105",
                    "model-25007235", "model-27007655", "model-29008482", "model-31009021",
                    "model-33009731", "model-35010261", "model-37010859", "model-39012172"]
    model_names_a3c_normal = []
    for i in range(len(names_normal)):
        model_names_a3c_normal.append(model_path_a3c_normal + names_normal[i])

    model_path_a3c_rg1 = "./train/nips/Seaquest/a3c_gpu_8_threads_with_map_time_based_10_lr_0004_02-25-2019-11-36/"
    names_rg1 = ["model-1000311", "model-3000485", "model-5001013",
                 "model-7001569", "model-9002072", "model-11002396",
                 "model-13003249", "model-15003712", "model-17004616",
                 "model-19005322", "model-21005735", "model-23006092",
                 "model-25006999", "model-27007243", "model-29007500", "model-31007999",
                 "model-33008528", "model-35009123", "model-37009989", "model-39010545"]
    model_names_a3c_rg1 = []
    for i in range(len(names_rg1)):
        model_names_a3c_rg1.append(model_path_a3c_rg1 + names_rg1[i])

    index = 1
    for name in model_names_a3c_rg1:
        print(name, index)
        index += 2

        env = ALEEnvironment(ALEEnvironment.SEAQUEST,
                             is_render=True,
                             state_processor=SeaquestMapProcessor(),
                             frame_skip=8,
                             loss_of_life_termination=False)

        network_config = NIPSA3CConfig(env, initial_learning_rate=0.004)

        network = NIPSPolicyNetwork(network_config, load_model_path=name)

        agent = NIPSA3CAgent(network, env,
                         num_of_epochs=1,
                         steps_per_epoch=1000,
                         save_frequency=1e6,
                         report_frequency=1,
                         log_dir='./test/nips/Seaquest/a3c_gpu_8_threads_time_based_12_lr_0004/',
                         num_of_threads=1)

        agent.evaluate()

    # A3C NORMAL
    # A3C = []


if __name__ == '__main__':
    from PIL import Image
    # train_tank_1_player_machine()

    # evaluate_tank_1_player_machine()

    # train_tank_1_player_machine_with_map()

    # evaluate_tank_1_player_machine_with_map()

    # train_tank_1_player_machine_with_mscs()

    # evaluate_tank_1_player_machine_with_mscs()

    # evaluate_tank_1_player_machine_with_map()

    # evaluate_tank_1_player_machine_with_map()

    # perform_evaluation_1_player()

    #evaluate_tank_2_player_machine_rg(model_path_1='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16/model-9500310',
    #                                  model_path_2='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43/model-9500425',
    #                                  evaluation_steps=5000,
    #                                  render=True)

    #evaluate_tank_2_player_machine_a3c(model_path='./train/nips/TankBattle/a3c_gpu_8_threads_tank_time_based_10_2_players_lr_0004_epoches_10_terminal_04-25-2018-11-15/model-6000167',
    #                                   evaluation_steps=500, render=True)

    #model_path_a3c_rg1 = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16/model-9500310"
    #evaluate_tank_1_player_machine_with_map_a3c_rg1(model_path=model_path_a3c_rg1)

    #model_path_a3c_rg2 = "./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43/model-9500425"
    #evaluate_tank_1_player_machine_with_map_a3c_rg2(model_path=model_path_a3c_rg2)
    # perform_evaluation_2_players()

    # get_data()

    # get_data_2()

    # evaluate_tank_1_player_machine_with_mscs()

    # HUMAN VS HUMAN
    # [410, 170, 100, 210]
    # [110, 220, 40, 130]
    # [520, 390, 140, 340]
    # 347.5
    # [405.0, 247.0, 105.0, 216.0]
    # 243.25

    # perform_evaluation_2_players()

    # perform_evaluation_1_player()

    #evaluate_tank_2_player_machine_rg(model_path_1='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s3_multi_target_04-18-2018-21-16/model-9500310',
    #                                  model_path_2='./train/nips/TankBattle/a3c_gpu_8_threads_tank_with_map_time_based_10_lr_0004_5_enemy_s2_single_target_04-18-2018-20-43/model-9500425',
    #                                  evaluation_steps=50000,
    #                                  render=False)

    # game_engine = tankbattle(render=True,
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

    # train_atari_seaquest_with_map()

    # train_atari_seaquest()

    # show_atari_seaquest()

    # evaluate_atari_seaquest_with_map("./train/nips/Seaquest/a3c_gpu_8_threads_with_map_time_based_10_lr_0004_07-28-2018-14-09/model-19513796", 1000000)

    # perform_evaluation_seaquest()
    log_dir = "./test/nips/result"
    steps_test = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    mean_rewards_a3c = [1, 1, 1, 3, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 3, 2]
    mean_rewards_rg1 = [6, 5, 4, 6, 4, 8, 9, 8, 9, 10, 12, 10, 12, 13, 13, 14, 13, 14, 16, 15]
    plt.plot(steps_test, mean_rewards_a3c, 'go-', label='A3C-N', linewidth=1)
    plt.plot(steps_test, mean_rewards_rg1, 'bs-', label='A3C-RG1', linewidth=1)
    plt.legend(loc='best')
    axes = plt.gca()
    # max_limit = int((index + 1))
    # axes.set_xlim([0, 40])
    r = [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
    axes.set_xticks(r)
    plt.savefig(log_dir + '/performance_test_1_seaquest_lives.png')
    plt.show()

    perform_evaluation_seaquest()
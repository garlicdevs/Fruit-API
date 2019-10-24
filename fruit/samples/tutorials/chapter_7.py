from scipy.stats import stats

from fruit.agents.factory import A3CAgent, MAA3CAgent, MapMAA3CAgent
from fruit.envs.games.milk_factory.engine import MilkFactory
from fruit.envs.juice import FruitEnvironment, MAFruitEnvironment
from fruit.networks.config.atari import AtariA3CConfig, AtariA3CMAConfig, AtariA3CMapMAConfig, AtariA3CFakeMapMAConfig, \
    AtariA3CTestMapMAConfig
from fruit.networks.policy import PolicyNetwork, MAPolicyNetwork, MapMAPolicyNetwork
from fruit.state.processor import AtariProcessor
import matplotlib.pyplot as plt


def train_milk_1_milk_1_fix_robots_with_no_status():

    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=1,
                              number_of_fix_robots=1, number_of_milks=1, seed=None, human_control=False, error_freq=0.03,
                              human_control_robot=0, milk_speed=3, debug=False, action_combined_mode=False, show_status=False)

    env = MAFruitEnvironment(game_engine,
                             max_episode_steps=200,
                             state_processor=AtariProcessor(),
                             multi_objective=False) # This will return single reward

    network_config = AtariA3CMAConfig(env, initial_learning_rate=0.001, beta=0.001)

    network = MAPolicyNetwork(network_config,
                              num_of_checkpoints=40,
                              using_gpu=True)

    agent = MAA3CAgent(network, env,
                       num_of_epochs=4,
                       steps_per_epoch=1e6,
                       save_frequency=1e5,
                       update_network_frequency=5,
                       log_dir='./train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_no_status_5_5_err_003',
                       num_of_threads=8)

    agent.train()


def train_milk_1_milk_1_fix_robots_with_status():

    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=1,
                              number_of_fix_robots=1, number_of_milks=1, seed=None, human_control=False, error_freq=0.03,
                              human_control_robot=0, milk_speed=3, debug=False, action_combined_mode=False, show_status=True)

    env = MAFruitEnvironment(game_engine,
                             max_episode_steps=200,
                             state_processor=AtariProcessor(),
                             multi_objective=False) # This will return single reward

    network_config = AtariA3CMAConfig(env, initial_learning_rate=0.001, beta=0.001)

    network = MAPolicyNetwork(network_config,
                              num_of_checkpoints=40,
                              using_gpu=True)

    agent = MAA3CAgent(network, env,
                       num_of_epochs=4,
                       steps_per_epoch=1e6,
                       save_frequency=1e5,
                       update_network_frequency=5,
                       log_dir='./train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_show_status_5_5_err_003',
                       num_of_threads=8)

    agent.train()


def train_milk_2_milk_1_fix_robots_with_no_status():

    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=2,
                              number_of_fix_robots=1, number_of_milks=2, seed=None, human_control=False, error_freq=0.01,
                              human_control_robot=0, milk_speed=3, debug=False, action_combined_mode=False, show_status=False,
                              number_of_exits=2)

    env = MAFruitEnvironment(game_engine,
                             max_episode_steps=200,
                             state_processor=AtariProcessor(),
                             multi_objective=False) # This will return single reward

    network_config = AtariA3CMAConfig(env, initial_learning_rate=0.001, beta=0.01)

    network = MAPolicyNetwork(network_config,
                              num_of_checkpoints=40,
                              using_gpu=True)

    agent = MAA3CAgent(network, env,
                       num_of_epochs=4,
                       steps_per_epoch=1e6,
                       save_frequency=1e5,
                       update_network_frequency=5,
                       log_dir='./train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_no_status_5_5_err_001',
                       num_of_threads=8)

    agent.train()


def train_milk_2_milk_1_fix_robots_with_status():

    game_engine = MilkFactory(render=False, speed=6000, max_frames=200, frame_skip=1, number_of_milk_robots=2,
                              number_of_fix_robots=1, number_of_milks=2, seed=None, human_control=False, error_freq=0.01,
                              human_control_robot=0, milk_speed=3, debug=False, action_combined_mode=False, show_status=True,
                              number_of_exits=2)

    env = MAFruitEnvironment(game_engine,
                             max_episode_steps=200,
                             state_processor=AtariProcessor(),
                             multi_objective=False) # This will return single reward

    network_config = AtariA3CMAConfig(env, initial_learning_rate=0.001, beta=0.01)

    network = MAPolicyNetwork(network_config,
                              num_of_checkpoints=40,
                              using_gpu=True)

    agent = MAA3CAgent(network, env,
                       num_of_epochs=4,
                       steps_per_epoch=1e6,
                       save_frequency=1e5,
                       update_network_frequency=5,
                       log_dir='./train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001',
                       num_of_threads=8)

    agent.train()


def evaluate(model, show_status=True, number_of_milks_robot=2, number_of_milks=2, error_freq=0.01, render=False, exits=2):

    num_of_threads = 8
    speed = 6000
    if render:
        num_of_threads = 1
        speed=20

    game_engine = MilkFactory(render=render, speed=speed, max_frames=200, frame_skip=1,
                              number_of_milk_robots=number_of_milks_robot,
                              number_of_fix_robots=1, number_of_milks=number_of_milks, seed=None,
                              human_control=False, error_freq=error_freq, number_of_exits=exits,
                              human_control_robot=0, milk_speed=3, debug=False, action_combined_mode=False,
                              show_status=show_status)

    env = MAFruitEnvironment(game_engine,
                             max_episode_steps=200,
                             state_processor=AtariProcessor(),
                             multi_objective=False) # This will return single reward

    network_config = AtariA3CMAConfig(env, initial_learning_rate=0.001, beta=0.001)

    network = MAPolicyNetwork(network_config,
                              num_of_checkpoints=40,
                              load_model_path=model,
                              using_gpu=True)

    agent = MAA3CAgent(network, env,
                       num_of_epochs=1,
                       steps_per_epoch=10000,
                       save_frequency=1e5,
                       update_network_frequency=5,
                       log_dir='./test/plos_one/milk/test',
                       num_of_threads=num_of_threads)

    score = agent.evaluate()

    print(score)
    return score


import numpy as np
import pickle
import os.path
colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6', '#e04e39']
markers = ['s-', 'o-', '^-', 'v-', 'x-', 'h-']


def report():
    # No visual map
    if os.path.isfile('./train/plos_one/results/1_1.b'):
        with open('./train/plos_one/results/1_1.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_no_status_5_5_05-22-2019-17-11/'
        models = ['model-100087', 'model-200113', 'model-300135', 'model-400201', 'model-500225',
                  'model-600304', 'model-700305', 'model-800368', 'model-900445',
                  'model-1000453', 'model-1100489', 'model-1200511', 'model-1300600',
                  'model-1400636', 'model-1500657', 'model-1600720', 'model-1700741',
                  'model-1800797', 'model-1900799', 'model-2000829', 'model-2100846',
                  'model-2200874', 'model-2300876', 'model-2400914', 'model-2500955',
                  'model-2600968', 'model-2701028', 'model-2801034', 'model-2901112',
                  'model-3001138', 'model-3101218', 'model-3201251', 'model-3301285',
                  'model-3401303', 'model-3501310', 'model-3601368', 'model-3701422',
                  'model-3801481', 'model-3901496']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=1, number_of_milks=1, error_freq=0.01, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_1.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[1], edgecolor=colors[1], label='A3C')

    # Show visual map
    if os.path.isfile('./train/plos_one/results/1_2.b'):
        with open('./train/plos_one/results/1_2.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_show_status_5_5_05-22-2019-17-11/'
        models = ['model-100031', 'model-200089', 'model-300116', 'model-400159', 'model-500185',
                  'model-600195', 'model-700269', 'model-800298', 'model-900302',
                  'model-1000376', 'model-1100417', 'model-1200502', 'model-1300513',
                  'model-1400512', 'model-1500581', 'model-1600589', 'model-1700655',
                  'model-1800737', 'model-1900777', 'model-2000826', 'model-2100867',
                  'model-2200908', 'model-2300927', 'model-2400947', 'model-2500981',
                  'model-2600991', 'model-2701004', 'model-2801020', 'model-2901106',
                  'model-3001189', 'model-3101273', 'model-3201320', 'model-3301390',
                  'model-3401418', 'model-3501494', 'model-3601524', 'model-3701574',
                  'model-3801593', 'model-3901674']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=1, number_of_milks=1, error_freq=0.01, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_2.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[3], edgecolor=colors[3], label='MVA3C')

    plt.xlabel('Training Steps (1e6)')
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0, 40)
    r = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    axes.set_xticks(r)

    plt.savefig('./train/plos_one/results/fig1.pdf')
    plt.show()


def report_2():
    # No visual map
    if os.path.isfile('./train/plos_one/results/1_3.b'):
        with open('./train/plos_one/results/1_3.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_no_status_5_5_err_001_05-25-2019-00-46/'
        models = ['model-100033', 'model-200076', 'model-300148', 'model-400239', 'model-500297',
                  'model-600383', 'model-700454', 'model-800516', 'model-900571',
                  'model-1000648', 'model-1100726', 'model-1200731', 'model-1300798',
                  'model-1400843', 'model-1500849', 'model-1600898', 'model-1700942',
                  'model-1800975', 'model-1900981', 'model-2000989', 'model-2101066',
                  'model-2201090', 'model-2301139', 'model-2401216', 'model-2501284',
                  'model-2601321', 'model-2701442', 'model-2801454', 'model-2901599',
                  'model-3001714', 'model-3101762', 'model-3201779', 'model-3301804',
                  'model-3401849', 'model-3501969', 'model-3602045', 'model-3702173',
                  'model-3802189', 'model-3902238']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=2, number_of_milks=2, error_freq=0.01, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_3.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[1], edgecolor=colors[1], label='A3C')

    # Show visual map
    if os.path.isfile('./train/plos_one/results/1_4.b'):
        with open('./train/plos_one/results/1_4.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001_05-25-2019-00-45/'
        models = ['model-100078', 'model-200136', 'model-300145', 'model-400220', 'model-500263',
                  'model-600325', 'model-700387', 'model-800423', 'model-900440',
                  'model-1000484', 'model-1100540', 'model-1200585', 'model-1300592',
                  'model-1400627', 'model-1500660', 'model-1600736', 'model-1700747',
                  'model-1800768', 'model-1900796', 'model-2000837', 'model-2100922',
                  'model-2200954', 'model-2300962', 'model-2401007', 'model-2501053',
                  'model-2601198', 'model-2701215', 'model-2801242', 'model-2901245',
                  'model-3001307', 'model-3101336', 'model-3201427', 'model-3301473',
                  'model-3401565', 'model-3501613', 'model-3601725', 'model-3701772',
                  'model-3801778', 'model-3901897']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=2, number_of_milks=2, error_freq=0.01, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_4.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[3], edgecolor=colors[3], label='MVA3C')

    plt.xlabel('Training Steps (1e5)')
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0, 40)
    r = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    axes.set_xticks(r)

    plt.savefig('./train/plos_one/results/fig2.pdf')
    plt.show()


def report_3():
    if os.path.isfile('./train/plos_one/results/1_5.b'):
        with open('./train/plos_one/results/1_5.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_show_status_5_5_05-22-2019-17-11/'
        models = ['model-100031', 'model-200089', 'model-300116', 'model-400159', 'model-500185',
                  'model-600195', 'model-700269', 'model-800298', 'model-900302',
                  'model-1000376', 'model-1100417', 'model-1200502', 'model-1300513',
                  'model-1400512', 'model-1500581', 'model-1600589', 'model-1700655',
                  'model-1800737', 'model-1900777', 'model-2000826', 'model-2100867',
                  'model-2200908', 'model-2300927', 'model-2400947', 'model-2500981',
                  'model-2600991', 'model-2701004', 'model-2801020', 'model-2901106',
                  'model-3001189', 'model-3101273', 'model-3201320', 'model-3301390',
                  'model-3401418', 'model-3501494', 'model-3601524', 'model-3701574',
                  'model-3801593', 'model-3901674']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=1, number_of_milks=1, error_freq=0.02, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_5.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[1], edgecolor=colors[1], label='ER = 0.02')

    if os.path.isfile('./train/plos_one/results/1_6.b'):
        with open('./train/plos_one/results/1_6.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_show_status_5_5_05-22-2019-17-11/'
        models = ['model-100031', 'model-200089', 'model-300116', 'model-400159', 'model-500185',
                  'model-600195', 'model-700269', 'model-800298', 'model-900302',
                  'model-1000376', 'model-1100417', 'model-1200502', 'model-1300513',
                  'model-1400512', 'model-1500581', 'model-1600589', 'model-1700655',
                  'model-1800737', 'model-1900777', 'model-2000826', 'model-2100867',
                  'model-2200908', 'model-2300927', 'model-2400947', 'model-2500981',
                  'model-2600991', 'model-2701004', 'model-2801020', 'model-2901106',
                  'model-3001189', 'model-3101273', 'model-3201320', 'model-3301390',
                  'model-3401418', 'model-3501494', 'model-3601524', 'model-3701574',
                  'model-3801593', 'model-3901674']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=1, number_of_milks=1, error_freq=0.03, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_6.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[3], edgecolor=colors[3], label='ER = 0.03')

    if os.path.isfile('./train/plos_one/results/1_7.b'):
        with open('./train/plos_one/results/1_7.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_show_status_5_5_05-22-2019-17-11/'
        models = ['model-100031', 'model-200089', 'model-300116', 'model-400159', 'model-500185',
                  'model-600195', 'model-700269', 'model-800298', 'model-900302',
                  'model-1000376', 'model-1100417', 'model-1200502', 'model-1300513',
                  'model-1400512', 'model-1500581', 'model-1600589', 'model-1700655',
                  'model-1800737', 'model-1900777', 'model-2000826', 'model-2100867',
                  'model-2200908', 'model-2300927', 'model-2400947', 'model-2500981',
                  'model-2600991', 'model-2701004', 'model-2801020', 'model-2901106',
                  'model-3001189', 'model-3101273', 'model-3201320', 'model-3301390',
                  'model-3401418', 'model-3501494', 'model-3601524', 'model-3701574',
                  'model-3801593', 'model-3901674']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=1, number_of_milks=1, error_freq=0.04, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_7.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[4], edgecolor=colors[4], label='ER = 0.04')

    if os.path.isfile('./train/plos_one/results/1_8.b'):
        with open('./train/plos_one/results/1_8.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_show_status_5_5_05-22-2019-17-11/'
        models = ['model-100031', 'model-200089', 'model-300116', 'model-400159', 'model-500185',
                  'model-600195', 'model-700269', 'model-800298', 'model-900302',
                  'model-1000376', 'model-1100417', 'model-1200502', 'model-1300513',
                  'model-1400512', 'model-1500581', 'model-1600589', 'model-1700655',
                  'model-1800737', 'model-1900777', 'model-2000826', 'model-2100867',
                  'model-2200908', 'model-2300927', 'model-2400947', 'model-2500981',
                  'model-2600991', 'model-2701004', 'model-2801020', 'model-2901106',
                  'model-3001189', 'model-3101273', 'model-3201320', 'model-3301390',
                  'model-3401418', 'model-3501494', 'model-3601524', 'model-3701574',
                  'model-3801593', 'model-3901674']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=1, number_of_milks=1,
                              error_freq=0.05, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_8.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1, 40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[5], edgecolor=colors[5], label='ER = 0.05')

    plt.xlabel('Training Steps (1e5)')
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0, 40)
    r = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    axes.set_xticks(r)

    plt.savefig('./train/plos_one/results/fig3.pdf')
    plt.show()


def report_4():
    if os.path.isfile('./train/plos_one/results/1_9.b'):
        with open('./train/plos_one/results/1_9.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001_05-25-2019-00-45/'
        models = ['model-100078', 'model-200136', 'model-300145', 'model-400220', 'model-500263',
                  'model-600325', 'model-700387', 'model-800423', 'model-900440',
                  'model-1000484', 'model-1100540', 'model-1200585', 'model-1300592',
                  'model-1400627', 'model-1500660', 'model-1600736', 'model-1700747',
                  'model-1800768', 'model-1900796', 'model-2000837', 'model-2100922',
                  'model-2200954', 'model-2300962', 'model-2401007', 'model-2501053',
                  'model-2601198', 'model-2701215', 'model-2801242', 'model-2901245',
                  'model-3001307', 'model-3101336', 'model-3201427', 'model-3301473',
                  'model-3401565', 'model-3501613', 'model-3601725', 'model-3701772',
                  'model-3801778', 'model-3901897']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.02, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_9.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[1], edgecolor=colors[1], label='ER = 0.02')

    if os.path.isfile('./train/plos_one/results/1_10.b'):
        with open('./train/plos_one/results/1_10.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001_05-25-2019-00-45/'
        models = ['model-100078', 'model-200136', 'model-300145', 'model-400220', 'model-500263',
                  'model-600325', 'model-700387', 'model-800423', 'model-900440',
                  'model-1000484', 'model-1100540', 'model-1200585', 'model-1300592',
                  'model-1400627', 'model-1500660', 'model-1600736', 'model-1700747',
                  'model-1800768', 'model-1900796', 'model-2000837', 'model-2100922',
                  'model-2200954', 'model-2300962', 'model-2401007', 'model-2501053',
                  'model-2601198', 'model-2701215', 'model-2801242', 'model-2901245',
                  'model-3001307', 'model-3101336', 'model-3201427', 'model-3301473',
                  'model-3401565', 'model-3501613', 'model-3601725', 'model-3701772',
                  'model-3801778', 'model-3901897']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.03, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_10.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[3], edgecolor=colors[3], label='ER = 0.03')

    if os.path.isfile('./train/plos_one/results/1_11.b'):
        with open('./train/plos_one/results/1_11.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001_05-25-2019-00-45/'
        models = ['model-100078', 'model-200136', 'model-300145', 'model-400220', 'model-500263',
                  'model-600325', 'model-700387', 'model-800423', 'model-900440',
                  'model-1000484', 'model-1100540', 'model-1200585', 'model-1300592',
                  'model-1400627', 'model-1500660', 'model-1600736', 'model-1700747',
                  'model-1800768', 'model-1900796', 'model-2000837', 'model-2100922',
                  'model-2200954', 'model-2300962', 'model-2401007', 'model-2501053',
                  'model-2601198', 'model-2701215', 'model-2801242', 'model-2901245',
                  'model-3001307', 'model-3101336', 'model-3201427', 'model-3301473',
                  'model-3401565', 'model-3501613', 'model-3601725', 'model-3701772',
                  'model-3801778', 'model-3901897']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.04, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_11.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[4], edgecolor=colors[4], label='ER = 0.04')

    if os.path.isfile('./train/plos_one/results/1_12.b'):
        with open('./train/plos_one/results/1_12.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001_05-25-2019-00-45/'
        models = ['model-100078', 'model-200136', 'model-300145', 'model-400220', 'model-500263',
                  'model-600325', 'model-700387', 'model-800423', 'model-900440',
                  'model-1000484', 'model-1100540', 'model-1200585', 'model-1300592',
                  'model-1400627', 'model-1500660', 'model-1600736', 'model-1700747',
                  'model-1800768', 'model-1900796', 'model-2000837', 'model-2100922',
                  'model-2200954', 'model-2300962', 'model-2401007', 'model-2501053',
                  'model-2601198', 'model-2701215', 'model-2801242', 'model-2901245',
                  'model-3001307', 'model-3101336', 'model-3201427', 'model-3301473',
                  'model-3401565', 'model-3501613', 'model-3601725', 'model-3701772',
                  'model-3801778', 'model-3901897']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=True, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.05, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_12.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1, 40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[5], edgecolor=colors[5], label='ER = 0.05')

    plt.xlabel('Training Steps (1e5)')
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0, 40)
    r = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    axes.set_xticks(r)

    plt.savefig('./train/plos_one/results/fig4.pdf')
    plt.show()


def report_5():
    if os.path.isfile('./train/plos_one/results/1_13.b'):
        with open('./train/plos_one/results/1_13.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_no_status_5_5_05-22-2019-17-11/'
        models = ['model-100087', 'model-200113', 'model-300135', 'model-400201', 'model-500225',
                  'model-600304', 'model-700305', 'model-800368', 'model-900445',
                  'model-1000453', 'model-1100489', 'model-1200511', 'model-1300600',
                  'model-1400636', 'model-1500657', 'model-1600720', 'model-1700741',
                  'model-1800797', 'model-1900799', 'model-2000829', 'model-2100846',
                  'model-2200874', 'model-2300876', 'model-2400914', 'model-2500955',
                  'model-2600968', 'model-2701028', 'model-2801034', 'model-2901112',
                  'model-3001138', 'model-3101218', 'model-3201251', 'model-3301285',
                  'model-3401303', 'model-3501310', 'model-3601368', 'model-3701422',
                  'model-3801481', 'model-3901496']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=1, number_of_milks=1, error_freq=0.02, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_13.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[1], edgecolor=colors[1], label='ER = 0.02')

    if os.path.isfile('./train/plos_one/results/1_14.b'):
        with open('./train/plos_one/results/1_14.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_no_status_5_5_05-22-2019-17-11/'
        models = ['model-100087', 'model-200113', 'model-300135', 'model-400201', 'model-500225',
                  'model-600304', 'model-700305', 'model-800368', 'model-900445',
                  'model-1000453', 'model-1100489', 'model-1200511', 'model-1300600',
                  'model-1400636', 'model-1500657', 'model-1600720', 'model-1700741',
                  'model-1800797', 'model-1900799', 'model-2000829', 'model-2100846',
                  'model-2200874', 'model-2300876', 'model-2400914', 'model-2500955',
                  'model-2600968', 'model-2701028', 'model-2801034', 'model-2901112',
                  'model-3001138', 'model-3101218', 'model-3201251', 'model-3301285',
                  'model-3401303', 'model-3501310', 'model-3601368', 'model-3701422',
                  'model-3801481', 'model-3901496']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=1, number_of_milks=1, error_freq=0.03, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_14.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[3], edgecolor=colors[3], label='ER = 0.03')

    if os.path.isfile('./train/plos_one/results/1_15.b'):
        with open('./train/plos_one/results/1_15.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_no_status_5_5_05-22-2019-17-11/'
        models = ['model-100087', 'model-200113', 'model-300135', 'model-400201', 'model-500225',
                  'model-600304', 'model-700305', 'model-800368', 'model-900445',
                  'model-1000453', 'model-1100489', 'model-1200511', 'model-1300600',
                  'model-1400636', 'model-1500657', 'model-1600720', 'model-1700741',
                  'model-1800797', 'model-1900799', 'model-2000829', 'model-2100846',
                  'model-2200874', 'model-2300876', 'model-2400914', 'model-2500955',
                  'model-2600968', 'model-2701028', 'model-2801034', 'model-2901112',
                  'model-3001138', 'model-3101218', 'model-3201251', 'model-3301285',
                  'model-3401303', 'model-3501310', 'model-3601368', 'model-3701422',
                  'model-3801481', 'model-3901496']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=1, number_of_milks=1, error_freq=0.04, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_15.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[4], edgecolor=colors[4], label='ER = 0.04')

    if os.path.isfile('./train/plos_one/results/1_16.b'):
        with open('./train/plos_one/results/1_16.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_no_status_5_5_05-22-2019-17-11/'
        models = ['model-100087', 'model-200113', 'model-300135', 'model-400201', 'model-500225',
                  'model-600304', 'model-700305', 'model-800368', 'model-900445',
                  'model-1000453', 'model-1100489', 'model-1200511', 'model-1300600',
                  'model-1400636', 'model-1500657', 'model-1600720', 'model-1700741',
                  'model-1800797', 'model-1900799', 'model-2000829', 'model-2100846',
                  'model-2200874', 'model-2300876', 'model-2400914', 'model-2500955',
                  'model-2600968', 'model-2701028', 'model-2801034', 'model-2901112',
                  'model-3001138', 'model-3101218', 'model-3201251', 'model-3301285',
                  'model-3401303', 'model-3501310', 'model-3601368', 'model-3701422',
                  'model-3801481', 'model-3901496']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=1, number_of_milks=1,
                              error_freq=0.05, exits=1)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_16.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1, 40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[5], edgecolor=colors[5], label='ER = 0.05')

    plt.xlabel('Training Steps (1e5)')
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0, 40)
    r = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    axes.set_xticks(r)

    plt.savefig('./train/plos_one/results/fig5.pdf')
    plt.show()


def report_6():
    if os.path.isfile('./train/plos_one/results/1_17.b'):
        with open('./train/plos_one/results/1_17.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_no_status_5_5_err_001_05-25-2019-00-46/'
        models = ['model-100033', 'model-200076', 'model-300148', 'model-400239', 'model-500297',
                  'model-600383', 'model-700454', 'model-800516', 'model-900571',
                  'model-1000648', 'model-1100726', 'model-1200731', 'model-1300798',
                  'model-1400843', 'model-1500849', 'model-1600898', 'model-1700942',
                  'model-1800975', 'model-1900981', 'model-2000989', 'model-2101066',
                  'model-2201090', 'model-2301139', 'model-2401216', 'model-2501284',
                  'model-2601321', 'model-2701442', 'model-2801454', 'model-2901599',
                  'model-3001714', 'model-3101762', 'model-3201779', 'model-3301804',
                  'model-3401849', 'model-3501969', 'model-3602045', 'model-3702173',
                  'model-3802189', 'model-3902238']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.02, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_17.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    plt.rc('grid', linestyle="--", color='gray', alpha=0.2)
    plt.grid(True)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[1])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[1], edgecolor=colors[1], label='ER = 0.02')

    if os.path.isfile('./train/plos_one/results/1_18.b'):
        with open('./train/plos_one/results/1_18.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_no_status_5_5_err_001_05-25-2019-00-46/'
        models = ['model-100033', 'model-200076', 'model-300148', 'model-400239', 'model-500297',
                  'model-600383', 'model-700454', 'model-800516', 'model-900571',
                  'model-1000648', 'model-1100726', 'model-1200731', 'model-1300798',
                  'model-1400843', 'model-1500849', 'model-1600898', 'model-1700942',
                  'model-1800975', 'model-1900981', 'model-2000989', 'model-2101066',
                  'model-2201090', 'model-2301139', 'model-2401216', 'model-2501284',
                  'model-2601321', 'model-2701442', 'model-2801454', 'model-2901599',
                  'model-3001714', 'model-3101762', 'model-3201779', 'model-3301804',
                  'model-3401849', 'model-3501969', 'model-3602045', 'model-3702173',
                  'model-3802189', 'model-3902238']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.03, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_18.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[3])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[3], edgecolor=colors[3], label='ER = 0.03')

    if os.path.isfile('./train/plos_one/results/1_19.b'):
        with open('./train/plos_one/results/1_19.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_no_status_5_5_err_001_05-25-2019-00-46/'
        models = ['model-100033', 'model-200076', 'model-300148', 'model-400239', 'model-500297',
                  'model-600383', 'model-700454', 'model-800516', 'model-900571',
                  'model-1000648', 'model-1100726', 'model-1200731', 'model-1300798',
                  'model-1400843', 'model-1500849', 'model-1600898', 'model-1700942',
                  'model-1800975', 'model-1900981', 'model-2000989', 'model-2101066',
                  'model-2201090', 'model-2301139', 'model-2401216', 'model-2501284',
                  'model-2601321', 'model-2701442', 'model-2801454', 'model-2901599',
                  'model-3001714', 'model-3101762', 'model-3201779', 'model-3301804',
                  'model-3401849', 'model-3501969', 'model-3602045', 'model-3702173',
                  'model-3802189', 'model-3902238']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.04, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_19.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1,40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[4])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[4], edgecolor=colors[4], label='ER = 0.04')

    if os.path.isfile('./train/plos_one/results/1_20.b'):
        with open('./train/plos_one/results/1_20.b', 'rb+') as file:
            d = pickle.load(file)
            mean_scores = d[0]
            errors = d[1]
    else:
        mean_scores = []
        errors = []
        base_path = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_no_status_5_5_err_001_05-25-2019-00-46/'
        models = ['model-100033', 'model-200076', 'model-300148', 'model-400239', 'model-500297',
                  'model-600383', 'model-700454', 'model-800516', 'model-900571',
                  'model-1000648', 'model-1100726', 'model-1200731', 'model-1300798',
                  'model-1400843', 'model-1500849', 'model-1600898', 'model-1700942',
                  'model-1800975', 'model-1900981', 'model-2000989', 'model-2101066',
                  'model-2201090', 'model-2301139', 'model-2401216', 'model-2501284',
                  'model-2601321', 'model-2701442', 'model-2801454', 'model-2901599',
                  'model-3001714', 'model-3101762', 'model-3201779', 'model-3301804',
                  'model-3401849', 'model-3501969', 'model-3602045', 'model-3702173',
                  'model-3802189', 'model-3902238']
        for model in models:
            print(model)
            scores = evaluate(base_path + model, show_status=False, number_of_milks_robot=2, number_of_milks=2,
                              error_freq=0.05, exits=2)
            ss = [s[0] for s in scores]
            mean_scores.append(np.mean(ss))
            errors.append(stats.sem(ss))

        with open('./train/plos_one/results/1_20.b', 'wb+') as file:
            pickle.dump([mean_scores, errors], file)

    y_ranges = range(1, 40)

    plt.plot(y_ranges, mean_scores, alpha=0.9, color=colors[5])
    plt.fill_between(y_ranges, np.array(mean_scores) - np.array(errors),
                     np.array(mean_scores) + np.array(errors),
                     alpha=0.1, facecolor=colors[5], edgecolor=colors[5], label='ER = 0.05')

    plt.xlabel('Training Steps (1e5)')
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim(0, 40)
    r = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    axes.set_xticks(r)

    plt.savefig('./train/plos_one/results/fig6.pdf')
    plt.show()


if __name__ == '__main__':
    # report()

    # model_1_1_001_no_status = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001_05-25-2019-00-45/model-3901897'
    # evaluate(model_1_1_001_no_status, render=True, number_of_milks=2, number_of_milks_robot=2, error_freq=0.01, show_status=True, exits=2)

    # model_1_1_001_with_status = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_1_fix_1_show_status_5_5_05-22-2019-17-11/model-3901674'
    # evaluate(model_1_1_001_with_status, render=True, number_of_milks=1, number_of_milks_robot=1, error_freq=0.05, show_status=True, exits=1)

    train_milk_2_milk_1_fix_robots_with_no_status()

    # model_2_1 = './train/plos_one/milk/a3c_gpu_8_threads_milk_fac_4_lr_0001_milk_2_fix_1_show_status_5_5_err_001_05-25-2019-00-45/model-3901897'
    # evaluate(model_2_1, render=True, number_of_milks_robot=1, number_of_milks=1, error_freq=0.01, show_status=True, exits=2)

    #report_2()
    #report()
    # report_2()
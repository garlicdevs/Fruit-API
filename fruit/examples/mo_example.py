from fruit.envs.juice import FruitEnvironment
from fruit.envs.games.deep_sea_treasure.engine import DeepSeaTreasure
from fruit.envs.games.mountain_car.engine import MountainCar
from fruit.agents.moq import MOQAgent
import matplotlib.pyplot as plt
from fruit.agents.moq import MODQNAgent
from fruit.networks.config.juice import MODQNConfig
from fruit.networks.config.juice import MOExDQNConfig
from fruit.networks.policy import MOPolicyNetwork
from threading import Thread
from fruit.utils.processor import AtariProcessor
import pickle
from fruit.examples.hypervolume import HVCalculator

########################################################
# Date: 18-Feb-2018
# Author: Duy Nguyen
# Email: garlicdevs@gmail.com
########################################################


def train_multi_objective_agent(env_size):

    # Create a Deep Sea Treasure game
    game = DeepSeaTreasure(width=env_size, seed=100)

    # Put game into fruit wrapper
    env = FruitEnvironment(game, multi_objective=True)

    # Create a multi-objective agent using Q-learning algorithm
    agent = MOQAgent(env, log_dir="./train/MORL/ql/deep_sea_treasure", steps_per_epoch=200000, num_of_epochs=1,
                     summary_frequency=190000)

    # Train it
    agent.train()


def train_multi_objective_agent_mountain_car():

    # Create a Mountain Car game
    game = MountainCar(graphical_state=False, frame_skip=1, render=False, speed=1000, is_debug=False)

    # Put game into fruit wrapper and enable multi-objective feature
    env = FruitEnvironment(game, multi_objective=True)

    # Create a multi-objective agent using Q-learning algorithm
    agent = MOQAgent(env, log_dir="./train/MORL/mountain_car/ql/mountain_car_0.5_0.3_0.2", steps_per_epoch=3000030,
                     num_of_epochs=1,
                     report_frequency=1, summary_frequency=100000,
                     is_linear = True,
                     thresholds = [0.5, 0.3, 0.2])
    print("Threshold:", [0.5, 0.3, 0.2])
    # Train the agent
    agent.train()


def multi_objective_agent_test():

    # Create a Deep Sea Treasure game
    game = DeepSeaTreasure(width=3, seed=100, render=True, speed=5)

    # Put game into fruit wrapper
    env = FruitEnvironment(game)

    # Create a multi-objective agent using Q-learning algorithm
    agent = MOQAgent(env, log_dir="./test/MORL/ql/deep_sea_treasure",
                     load_model_path="./train/MORL/ql/deep_sea_treasure_03-15-2018-12-26/checkpoint_190102.npy",
                     num_of_epochs=1, steps_per_epoch=100,
                     using_e_greedy=True, epsilon_annealing_start=0.,
                     )

    # Train it
    best_reward = agent.evaluate()

    # Print the best reward
    print("Best reward:", best_reward[0][0])


def multi_objective_agent_mountain_car_test():

    # Create a Mountain Car game
    game = MountainCar(graphical_state=False, frame_skip=1, render=True, speed=1000, is_debug=False)

    # Put game into fruit wrapper and enable multi-objective feature
    env = FruitEnvironment(game, multi_objective=True)

    # Create a multi-objective agent using Q-learning algorithm
    agent = MOQAgent(env, log_dir="./test/MORL/ql/mountain_car",
                     load_model_path="./train/MORL/ql/mountain_car_1_1_0_05-24-2018-22-53/checkpoint_2400019.npy",
                     steps_per_epoch=10000, num_of_epochs=1,
                     using_e_greedy=True, epsilon_annealing_start=0.,
                     report_frequency=1,
                     is_linear = True,
                     thresholds = [1, 1, 0])

    # Evaluate the agent
    agent.evaluate()


def additive_epsilon(this_reward, target):
    score = 0.
    for i in range(len(target)):
        diff = target[i] - this_reward[i]
        if diff > score:
            score = diff
    return score


def additive_epsilon_history(pareto_set, agent_set, num_of_agent_solutions):
    max_score = 0.
    for i in range(len(pareto_set)):
        min_score = 5000
        for j in range(num_of_agent_solutions):
            score = additive_epsilon(agent_set[j], pareto_set[i])
            if score < min_score:
                min_score = score
        if min_score > max_score:
            max_score = min_score
    return max_score


def mo_ql_test():
    envs = [3, 5, 7]
    stat = [[] for _ in range(len(envs))]
    total_steps = [0 for _ in range(len(envs))]

    for e in range(len(envs)):
        # Create game
        game = DeepSeaTreasure(width=envs[e], seed=100)

        # Put game into Fruit wrapper
        env = FruitEnvironment(game)

        # Get pareto set of this game
        pareto_set = env.get_pareto_solutions()

        # Prepare thresholds
        num_of_objs = env.get_num_of_objectives()
        thresholds = [0] * (num_of_objs-1)

        # Run the training process for each pareto solution
        agent_set = pareto_set
        for i in range(len(pareto_set)):
            steps = total_steps[e]

            # Calculate threshold
            if i == 0:
                for j in range(len(thresholds)):
                    thresholds[j] = pareto_set[i][j] - 10
            else:
                for j in range(len(thresholds)):
                    thresholds[j] = 0.5 * (pareto_set[i][j] + pareto_set[i-1][j])

            # Train agent
            print(thresholds)
            print(pareto_set[i])
            agent = MOQAgent(env, log_dir="./train/MORL/Deep_sea_treasure", thresholds=thresholds,
                             target_reward=pareto_set[i])
            reward_list = agent.train()

            for j in reward_list:
                agent_set[i] = j[0]
                score = additive_epsilon_history(pareto_set, agent_set, i + 1)
                stat[e].append([score, steps + j[1]])

            total_steps[e] = total_steps[e] + reward_list[-1][1]

    plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'go-', label='Env Width = 3', linewidth=1)
    plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'rs-', label='Env Width = 5', linewidth=1)
    plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='Env Width = 7', linewidth=1)
    plt.savefig("./test/MORL/deepseatreasure_performance_test.png")
    plt.legend(loc='best')
    plt.show()


def train_mo_dqn_agent(env_size=5, config=0, reward_list=None, extended_config=True, is_linear=True, parallel=False):

    # Create a Deep Sea Treasure game
    game = DeepSeaTreasure(graphical_state=True, width=env_size, seed=100, render=False, max_treasure=100, speed=1000)

    # Put game into fruit wrapper
    env = FruitEnvironment(game,
                           max_episode_steps=60,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    # Get treasures
    treasures = game.get_treasure()

    is_linear = is_linear

    if is_linear:
        TLO_thresholds = None
        linear_thresholds = None
        if env_size == 3:
            if config == 0:
                linear_thresholds = [0.01, 0.99]
            elif config == 1:
                linear_thresholds = [0.36, 0.64]
            else:
                linear_thresholds = [1, 0]
        elif env_size == 5:
            if config == 0:
                linear_thresholds = [0.01, 0.99]
            elif config == 1:
                linear_thresholds = [0.08, 0.92]
            elif config == 2:
                linear_thresholds = [0.3, 0.7]
            elif config == 3:
                linear_thresholds = [0.7, 0.3]
            else:
                linear_thresholds = [1, 0]
        elif env_size == 7:
            if config == 0:
                linear_thresholds = [0.01, 0.99]
            elif config == 1:
                linear_thresholds = [0.05, 0.95]
            elif config == 2:
                linear_thresholds = [0.26, 0.74]
            elif config == 3:
                linear_thresholds = [0.47, 0.53]
            elif config == 4:
                linear_thresholds = [0.77, 0.23]
            else:
                linear_thresholds = [1, 0]
    else:
        linear_thresholds = [10, 1]
        if config == 0:
            TLO_thresholds = [0]
        else:
            TLO_thresholds = [(treasures[config]+treasures[config-1])/2]

    print("Config:", is_linear, linear_thresholds, TLO_thresholds)
    if extended_config:
        config = MOExDQNConfig(env, is_linear=is_linear, linear_thresholds=linear_thresholds,
                             TLO_thresholds=TLO_thresholds, num_of_objectives=2, using_cnn=True, history_length=4)
    else:
        config = MODQNConfig(env, is_linear=is_linear, linear_thresholds=linear_thresholds,
                             TLO_thresholds=TLO_thresholds)

    # Create a shared policy network
    network = MOPolicyNetwork(config, num_of_checkpoints=10)

    # Create a multi-objective agent using Q-learning algorithm
    steps_per_epoch = 100000
    if env_size == 3:
        steps_per_epoch = 10000
        num_of_epochs = 5
        warmup_steps = 5000
        epsilon_annealing_steps = num_of_epochs * steps_per_epoch-4000
        exp_replay_size = 50000
        if parallel:
            report_frequency = 100
        else:
            report_frequency = 200

        if extended_config:
            if parallel:
                log_dir = "./train/MORL/deep_sea_treasure/hypervolume/parallel/env_3/Deep_sea_treasure"
            else:
                log_dir = "./train/MORL/deep_sea_treasure/hypervolume/multiple/env_3/Deep_sea_treasure"
        else:
            log_dir = "./train/MORL/deep_sea_treasure/hypervolume/single/env_3/Deep_sea_treasure"
    elif env_size == 5:
        num_of_epochs = 2
        warmup_steps = 10000
        epsilon_annealing_steps = num_of_epochs * steps_per_epoch-10000
        exp_replay_size = 100000
        if parallel:
            report_frequency = 250
        else:
            report_frequency = 500
        if extended_config:
            if parallel:
                log_dir = "./train/MORL/deep_sea_treasure/hypervolume/parallel/env_5/Deep_sea_treasure"
            else:
                log_dir = "./train/MORL/deep_sea_treasure/hypervolume/multiple/env_5/Deep_sea_treasure"
        else:
            log_dir = "./train/MORL/deep_sea_treasure/hypervolume/single/env_5/Deep_sea_treasure"
    else:
        num_of_epochs = 20
        warmup_steps = 30000
        epsilon_annealing_steps = num_of_epochs * steps_per_epoch - 50000
        exp_replay_size = 300000
        if parallel:
            report_frequency = 1250
        else:
            report_frequency = 2500
        if extended_config:
            if parallel:
                log_dir = "./train/MORL/deep_sea_treasure/hypervolume/parallel/env_7/Deep_sea_treasure"
            else:
                log_dir = "./train/MORL/deep_sea_treasure/hypervolume/multiple/env_7/Deep_sea_treasure"
        else:
            log_dir = "./train/MORL/deep_sea_treasure/hypervolume/single/env_7/Deep_sea_treasure"

    agent = MODQNAgent(network, env, log_dir=log_dir, num_of_epochs=num_of_epochs, report_frequency=report_frequency,
                       steps_per_epoch=steps_per_epoch, warmup_steps=warmup_steps, final_epsilon=0.,
                       epsilon_anneal_steps=epsilon_annealing_steps, exp_replay_size=exp_replay_size)

    # Train it
    reward_l = agent.train()
    if reward_list is not None:
        reward_list.append(reward_l)
    return reward_l


def train_mo_dqn_agent_mountain_car(config, reward_list=None, extended_config=False, using_cnn=False,
                                    is_linear=True, parallel=False):

    # Create a Mountain Car game
    if using_cnn:
        game = MountainCar(graphical_state=True, frame_skip=5, render=False, speed=1000, is_debug=False)
    else:
        game = MountainCar(graphical_state=False, frame_skip=1, render=False, speed=1000, is_debug=False)

    # Put game into fruit wrapper
    env = FruitEnvironment(game,
                           max_episode_steps=100,
                           state_processor=AtariProcessor(),
                           multi_objective=True)

    # Create a configuration network
    if is_linear:
        is_linear = True
        TLO_thresholds = None
        linear_thresholds = None
        if config == 0:
            linear_thresholds = [1, 0, 0]
        elif config == 1:
            linear_thresholds = [0.5, 0.5, 0]
        elif config == 2:
            linear_thresholds = [0.5, 0, 0.5]
        elif config == 3:
            linear_thresholds = [0, 1, 0]
        elif config == 4:
            linear_thresholds = [0, 0.5, 0.5]
        elif config == 5:
            linear_thresholds = [0, 0, 1]
    else:
        is_linear = False
        linear_thresholds = [10, 10, 10]
        TLO_thresholds = None
        if config == 0:
            linear_thresholds = [10, 10, 0] # [1, 0, 0]
            TLO_thresholds = [0, -110]
        elif config == 1:
            linear_thresholds = [10, 10, 0] # [0, 1, 0]
            TLO_thresholds = [-110, 0]
        elif config == 2:
            TLO_thresholds = [-110, -110] # [0, 0, 1]
        elif config == 3:
            linear_thresholds = [10, 10, 0]
            TLO_thresholds = [-5, -3] # [0.5, 0.5, 0]
        elif config == 4:
            TLO_thresholds = [-5, -110] # [0.5, 0, 0.5]
        elif config == 5:
            TLO_thresholds = [-110, -3] # [0, 0.5, 0.5]

    print("Config:", is_linear, linear_thresholds, TLO_thresholds)
    if extended_config:
        if using_cnn:
            config = MOExDQNConfig(env, is_linear=is_linear, linear_thresholds=linear_thresholds,
                             TLO_thresholds=TLO_thresholds, num_of_objectives=3, using_cnn=True, history_length=4)
        else:
            config = MOExDQNConfig(env, is_linear=is_linear, linear_thresholds=linear_thresholds,
                                   TLO_thresholds=TLO_thresholds, num_of_objectives=3, using_cnn=False)
    else:
        config = MODQNConfig(env, is_linear=is_linear, linear_thresholds=linear_thresholds,
                             TLO_thresholds=TLO_thresholds, num_of_objectives=3)

    # Create a shared policy network
    network = MOPolicyNetwork(config)

    # Create a multi-objective agent using Q-learning algorithm
    steps_per_epoch = 100000
    num_of_epochs = 2
    warmup_steps = 2000
    epsilon_annealing_steps = 200000
    exp_replay_size = 20000
    if extended_config:
        if parallel:
            log_dir = "./train/MORL/mountain_car/parallel/mountain_car"
        else:
            log_dir = "./train/MORL/mountain_car/multiple/mountain_car"
    else:
        log_dir = "./train/MORL/mountain_car/single/mountain_car"

    agent = MODQNAgent(network, env, log_dir=log_dir, num_of_epochs=num_of_epochs, initial_epsilon=1.,
                       steps_per_epoch=steps_per_epoch, warmup_steps=warmup_steps, final_epsilon=0.,
                       epsilon_anneal_steps=epsilon_annealing_steps, exp_replay_size=exp_replay_size,
                       report_frequency=20)

    # Train it
    reward_l = agent.train()
    if reward_list is not None:
        reward_list.append(reward_l)
    return reward_l


def mo_dqn_single_policy_test_distance(extended_config=False, env_width=3):
    envs = [env_width]
    stat = [[] for _ in range(len(envs))]
    total_steps = [0 for _ in range(len(envs))]

    for e in range(len(envs)):
        # Create game
        game = DeepSeaTreasure(width=envs[e], seed=100, max_treasure=100)

        # Put game into Fruit wrapper
        env = FruitEnvironment(game)

        # Get pareto set of this game
        pareto_set = env.get_pareto_solutions()

        # Run the training process for each pareto solution
        agent_set = pareto_set
        for i in range(len(pareto_set)):
            steps = total_steps[e]

            print("Pareto solution:", pareto_set[i])

            reward_list = train_mo_dqn_agent(env_size=envs[e], config=i, extended_config=extended_config)

            for j in reward_list:
                agent_set[i] = j[0]
                score = additive_epsilon_history(pareto_set, agent_set, i + 1)
                stat[e].append([score, steps + j[1]])

            total_steps[e] = total_steps[e] + reward_list[-1][1]

    if envs[0] == 3:
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'go-', label='Env Width = 3', linewidth=1)
    elif envs[0] == 5:
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'rs-', label='Env Width = 5', linewidth=1)
    else:
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'bx-', label='Env Width = 7', linewidth=1)
    if extended_config:
        plt.savefig("./test/MORL/multiple/deepseatreasure_dqn_performance_test_env_"+ str(envs[0]) +".png")
    else:
        plt.savefig("./test/MORL/single/deepseatreasure_dqn_performance_test_env_" + str(envs[0]) + ".png")
    plt.legend(loc='best')
    plt.show()


def calculate_hypervolume(reward_list, reference_point, min_num):
    num = len(reward_list)

    rewards = []
    for i in range(min_num):
        r = []
        for j in range(num):
            r.append(reward_list[j][i])
        rewards.append(r)

    vols = []

    for i in range(len(rewards)):
        r1 = rewards[i]
        h = 0
        r = []
        for j in range(len(r1)):
            x_p = r1[j][1]
            y_p = r1[j][0]

            x_r = reference_point[1]
            y_r = reference_point[0]

            if x_p <= x_r or y_p <= y_r:
                continue
            else:
                r.append([y_p, x_p])

        r = sorted(r)

        for j in range(len(r)):
            x_p = r[j][1]
            y_p = r[j][0]

            x_r = reference_point[1]
            y_r = reference_point[0]

            if j == 0:
                h = h + (x_p - x_r) * (y_p - y_r)

            elif j > 0:
                y_r = r[j-1][0]
                x_r = r[j-1][1]

                if x_p >= x_r and y_p >= y_r:
                    # print(".", x_p, x_r, y_p, y_r, (x_p - x_r) * (y_p - y_r), (x_p - x_r)*(y_p - reference_point[0]),
                    # (x_p - reference_point[1])*(y_p - y_r))
                    h = h + (x_p - x_r) * (y_p - y_r) + (x_p - x_r)*(y_r - reference_point[0]) + (x_r - reference_point[1])*(y_p - y_r)
                elif x_p >= x_r and y_p <= y_r:
                    # print("..", x_p, x_r, y_p, y_r)
                    h = h + (x_p - x_r)*(y_p - reference_point[0])
                elif x_p <= x_r and y_p >= y_r:
                    # print("...", x_p, x_r, y_p, y_r)
                    h = h + (x_p - reference_point[1])*(y_p - y_r)

        vols.append(h)

    return vols


def mo_dqn_single_policy_test_hypervolume(extended_config=True, env_width=3):
    envs = [env_width]
    stat = [[] for _ in range(3)]
    total_steps = [0 for _ in range(len(envs))]
    if env_width == 3:
        reference_point = [-20, -10]
    elif env_width == 5:
        reference_point = [-20, -20]
    else:
        reference_point = [-20, -25]

    for e in range(len(envs)):
        # Create game
        game = DeepSeaTreasure(width=envs[e], seed=100, max_treasure=100)

        # Put game into Fruit wrapper
        env = FruitEnvironment(game)

        # Get pareto set of this game
        pareto_set = env.get_pareto_solutions()

        # Run the training process for each pareto solution
        agent_set = [[reference_point] for _ in range(env_width)]

        for i in range(len(pareto_set)):
            print("Pareto solution:", pareto_set[i])
            steps = total_steps[e]

            reward_list = train_mo_dqn_agent(env_size=envs[e], config=i, extended_config=extended_config,
                                             is_linear=True)

            for j in reward_list:
                agent_set[i] = [j[0]]
                vol = calculate_hypervolume(agent_set, reference_point, 1)
                print("VOL:", vol, agent_set)
                if vol[0] > 494.5:
                    print(agent_set, reference_point)
                stat[e].append([vol, steps + j[1]])

            total_steps[e] = total_steps[e] + reward_list[-1][1]

            print(reward_list)

    total_steps = [0 for _ in range(len(envs))]

    for e in range(len(envs)):
        # Create game
        game = DeepSeaTreasure(width=envs[e], seed=100, max_treasure=100)

        # Put game into Fruit wrapper
        env = FruitEnvironment(game)

        # Get pareto set of this game
        pareto_set = env.get_pareto_solutions()

        # Run the training process for each pareto solution
        agent_set = [[reference_point] for _ in range(env_width)]

        for i in range(len(pareto_set)):
            print("Pareto solution:", pareto_set[i])
            a = []
            steps = total_steps[e]

            reward_list = train_mo_dqn_agent(env_size=envs[e], config=i, extended_config=extended_config,
                                             is_linear=False)
            for j in reward_list:
                agent_set[i] = [j[0]]
                vol = calculate_hypervolume(agent_set, reference_point, 1)
                if vol[0] > 494.5:
                    print(agent_set, reference_point)
                stat[2].append([vol, steps + j[1]])
                if envs[0] == 3:
                    stat[1].append([494.5, steps + j[1]])
                elif envs[0] == 5:
                    stat[1].append([1251.67, steps + j[1]])
                elif envs[0] == 7:
                    stat[1].append([1851.67, steps + j[1]])

            total_steps[e] = total_steps[e] + reward_list[-1][1]

            print(reward_list)

    if envs[0] == 3:
        with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_3/data_hypervolume.bin', 'wb') as file:
            pickle.dump(stat, file)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    elif envs[0] == 5:
        with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_5/data_hypervolume.bin', 'wb') as file:
            pickle.dump(stat, file)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    else:
        with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_7/data_hypervolume.bin', 'wb') as file:
            pickle.dump(stat, file)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    plt.legend(loc='best')
    if extended_config:
        plt.savefig("./test/MORL/deep_sea_treasure/hypervolume/multiple/deepseatreasure_dqn_performance_test_env_"+ str(envs[0]) +".pdf")
    else:
        plt.savefig("./test/MORL/deep_sea_treasure/hypervolume/single/deepseatreasure_dqn_performance_test_env_" + str(envs[0]) + ".pdf")
    plt.show()


def mo_dqn_single_policy_test_hypervolume_mountain_car(extended_config=True, env_width=6):
    envs = [env_width]
    stat = [[] for _ in range(3)]
    total_steps = [0 for _ in range(len(envs))]
    reference_point = [-110, -60, -60]

    for e in range(len(envs)):

        # Run the training process for each pareto solution
        agent_set = [reference_point for _ in range(env_width)]

        for i in range(len(agent_set)):
            steps = total_steps[e]

            reward_list = train_mo_dqn_agent_mountain_car(config=i, extended_config=extended_config,
                                             using_cnn=True, is_linear=True, parallel=False)

            for j in reward_list:
                agent_set[i] = j[0]
                vol = HVCalculator.get_volume_from_array(agent_set, reference_point)
                print("VOL:", vol, agent_set)
                stat[e].append([vol, steps + j[1]])

            total_steps[e] = total_steps[e] + reward_list[-1][1]

            print(reward_list)

    total_steps = [0 for _ in range(len(envs))]

    for e in range(len(envs)):

        # Run the training process for each pareto solution
        agent_set = [reference_point for _ in range(env_width)]

        for i in range(len(agent_set)):
            steps = total_steps[e]

            reward_list = train_mo_dqn_agent_mountain_car(config=i, extended_config=extended_config,
                                             using_cnn=True, is_linear=False, parallel=False)
            for j in reward_list:
                agent_set[i] = j[0]
                vol = HVCalculator.get_volume_from_array(agent_set, reference_point)

                stat[2].append([vol, steps + j[1]])
            total_steps[e] = total_steps[e] + reward_list[-1][1]

            print(reward_list)

    with open('./train/MORL/mountain_car/multiple/data_hypervolume2.bin', 'wb') as file:
        pickle.dump(stat, file)
    plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
    plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO MO-DQN', linewidth=1)
    # plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    plt.legend(loc='best')
    if extended_config:
        plt.savefig("./test/MORL/mountain_car/multiple/mc_dqn_performance_test_env2_"+ str(envs[0]) +".pdf")
    else:
        plt.savefig("./test/MORL/mountain_car/single/mc_dqn_performance_test_env2_" + str(envs[0]) + ".pdf")
    plt.show()


def average(stat, times):
    for j in range(times):
        for i in range(len(stat)-1):
            vol = stat[i][0]
            if type(vol)==list:
                vol = vol[0]
            next_vol = stat[i+1][0]
            if type(next_vol)==list:
                next_vol = next_vol[0]
            vol = (vol + next_vol)/2
            if type(stat[i][0]) == list:
                stat[i][0] = [vol]
            else:
                stat[i][0] = vol


def get_data_combined(times=15, env_width=3):
    envs = [env_width]
    if envs[0] == 3:
        with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_3/data_hypervolume.bin', 'rb') as file:
            stat_p = pickle.load(file)
        with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_3/data_hypervolume.bin', 'rb') as file:
            stat_m = pickle.load(file)

        average(stat_p[0], times)
        average(stat_p[2], times)
        average(stat_m[0], times)
        average(stat_m[2], times)

        plt.plot([x[1] for x in stat_m[1]], [x[0] for x in stat_m[1]], 'g--', label='Pareto', linewidth=1)
        plt.plot([x[1] for x in stat_p[0]], [x[0] for x in stat_p[0]], 'r.-', label='Multi-Policy Linear', linewidth=1)
        plt.plot([x[1] for x in stat_p[2]], [x[0] for x in stat_p[2]], 'bx-', label='Multi-Policy Non-Linear', linewidth=1)
        plt.plot([x[1] for x in stat_m[0]], [x[0] for x in stat_m[0]], 'c|-', label='Single-Policy Linear', linewidth=1)
        plt.plot([x[1] for x in stat_m[2]], [x[0] for x in stat_m[2]], 'm^-', label='Single-Policy Non-Linear', linewidth=1)

    elif envs[0] == 5:
        with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_5/data_hypervolume.bin', 'rb') as file:
            stat_p = pickle.load(file)
        with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_5/data_hypervolume.bin', 'rb') as file:
            stat_m = pickle.load(file)

        average(stat_p[0], times)
        average(stat_p[2], times)

        average(stat_m[0], times)
        average(stat_m[2], times)

        plt.plot([x[1] for x in stat_m[1]], [x[0] for x in stat_m[1]], 'g--', label='Pareto', linewidth=1)
        plt.plot([x[1] for x in stat_p[0]], [x[0] for x in stat_p[0]], 'r.-', label='Multi-Policy Linear', linewidth=1)
        plt.plot([x[1] for x in stat_p[2]], [x[0] for x in stat_p[2]], 'bx-', label='Multi-Policy Non-Linear', linewidth=1)
        plt.plot([x[1] for x in stat_m[0]], [x[0] for x in stat_m[0]], 'c|-', label='Single-Policy Linear', linewidth=1)
        plt.plot([x[1] for x in stat_m[2]], [x[0] for x in stat_m[2]], 'm^-', label='Single-Policy Non-Linear',
                 linewidth=1)

    plt.legend(loc='best')

    plt.savefig(
            "./test/MORL/deep_sea_treasure/hypervolume/parallel/deepseatreasure_dqn_performance_test_combined_env_" + str(
                envs[0]) + ".pdf")
    plt.show()


def get_data(parallel=False, times=2, env_width=3):
    envs = [env_width]
    if envs[0] == 3:
        if parallel:
            with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_3/data_hypervolume.bin', 'rb') as file:
                stat = pickle.load(file)
        else:
            with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_3/data_hypervolume.bin', 'rb') as file:
                stat = pickle.load(file)

        average(stat[0], times)
        average(stat[2], times)

        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='Non-Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    elif envs[0] == 5:
        if parallel:
            with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_5/data_hypervolume.bin', 'rb') as file:
                stat = pickle.load(file)
        else:
            with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_5/data_hypervolume.bin', 'rb') as file:
                stat = pickle.load(file)
        average(stat[0], times)
        average(stat[2], times)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='Non-Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    else:
        if parallel:
            with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_7/data_hypervolume.bin', 'rb') as file:
                stat = pickle.load(file)
        else:
            with open('./train/MORL/deep_sea_treasure/hypervolume/multiple/env_7/data_hypervolume.bin', 'rb') as file:
                stat = pickle.load(file)
        average(stat[0], times)
        average(stat[2], times)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='Non-Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)

    plt.legend(loc='best')
    if parallel:
        plt.savefig(
            "./test/MORL/deep_sea_treasure/hypervolume/parallel/deepseatreasure_dqn_performance_test_env_1_" + str(
                envs[0]) + ".pdf")
    else:
        plt.savefig(
            "./test/MORL/deep_sea_treasure/hypervolume/multiple/deepseatreasure_dqn_performance_test_env_1_" + str(
                envs[0]) + ".pdf")
    plt.show()


def get_data_mc_combined(times=15, env_width=6, skip=2):
    envs = [env_width]
    with open('./train/MORL/mountain_car/parallel/data_hypervolume2.bin', 'rb') as file:
        stat_p = pickle.load(file)
    with open('./train/MORL/mountain_car/multiple/data_hypervolume2.bin', 'rb') as file:
        stat_m = pickle.load(file)

    temp = [[], [], []]
    print(stat_p[0])
    for i in range(len(stat_p[0])):
        if i % (skip) == 0:
            temp[0].append(stat_p[0][i])

    for i in range(len(stat_p[2])):
        if i % (skip) == 0:
            temp[2].append(stat_p[2][i])

    stat_p = temp

    temp = [[], [], []]
    print(stat_m[0])
    for i in range(len(stat_m[0])):
        if i % (skip+6) == 0:
            temp[0].append(stat_m[0][i])

    for i in range(len(stat_m[2])):
        if i % (skip+6) == 0:
            temp[2].append(stat_m[2][i])

    stat_m = temp

    average(stat_p[0], times)
    average(stat_p[2], times)

    average(stat_m[0], times)
    average(stat_m[2], times)

    plt.plot([x[1] for x in stat_p[0]], [x[0] for x in stat_p[0]], 'r.-', label='Multi-Policy Linear', linewidth=1)
    plt.plot([x[1] for x in stat_p[2]], [x[0] for x in stat_p[2]], 'bx-', label='Multi-Policy Non-Linear', linewidth=1)

    plt.plot([x[1] for x in stat_m[0]], [x[0] for x in stat_m[0]], 'c|-', label='Single-Policy Linear', linewidth=1)
    plt.plot([x[1] for x in stat_m[2]], [x[0] for x in stat_m[2]], 'm^-', label='Single-Policy Non-Linear', linewidth=1)

    plt.legend(bbox_to_anchor=(0.51, 0.7), loc=2, borderaxespad=0.)

    plt.savefig(
            "./test/MORL/mountain_car/parallel/mc_dqn_performance_test_combined_env_" + str(
                envs[0]) + ".pdf")

    plt.show()


def get_data_mc(parallel=False, times=20, env_width=6, skip=7):
    envs = [env_width]
    if parallel:
        with open('./train/MORL/mountain_car/parallel/data_hypervolume2.bin', 'rb') as file:
            stat = pickle.load(file)
    else:
        with open('./train/MORL/mountain_car/multiple/data_hypervolume2.bin', 'rb') as file:
            stat = pickle.load(file)

    temp = [[], [], []]
    print(stat[0])
    for i in range(len(stat[0])):
        if i % skip == 0:
            temp[0].append(stat[0][i])

    for i in range(len(stat[2])):
        if i % skip == 0:
            temp[2].append(stat[2][i])

    stat = temp

    average(stat[0], times)
    average(stat[2], times)

    plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
    plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='Non-Linear MO-DQN', linewidth=1)

    plt.legend(loc='best')
    if parallel:
        plt.savefig(
            "./test/MORL/mountain_car/parallel/mc_dqn_performance_test_env_1_" + str(
                envs[0]) + ".pdf")
    else:
        plt.savefig(
            "./test/MORL/mountain_car/multiple/mc_dqn_performance_test_env_1_" + str(
                envs[0]) + ".pdf")
    plt.show()


def mo_dqn_multiple_policy_test_hypervolume(extended_config=True, env_width=3):
    envs = [env_width]
    stat = [[] for _ in range(3)]
    total_steps = []
    if env_width == 3:
        reference_point = [-20, -10]
    elif env_width == 5:
        reference_point = [-20, -20]
    else:
        reference_point = [-20, -25]

    policies = []
    reward_list = []

    # Run the training process for each pareto solution
    agent_set = [[] for _ in range(env_width)]

    for i in range(env_width):
        policies.append(Thread(target=train_mo_dqn_agent, args=(env_width, i, reward_list, extended_config, True, True)))
        policies[i].start()

    for i in range(env_width):
        policies[i].join()

    min = 100000
    min_index = -1
    for i in range(env_width):
        print(reward_list[i])
        if len(reward_list[i]) < min:
            min = len(reward_list[i])
            min_index = i

    for i in range(len(reward_list[min_index])):
        training_step = reward_list[min_index][i][1]
        agent_set[min_index].append(reward_list[min_index][i][0])
        total_steps.append(reward_list[min_index][i][1])
        for j in range(env_width):
            if j != min_index:
                for k in range(len(reward_list[j])):
                    if reward_list[j][k][1] >= training_step:
                        if k > 0:
                            agent_set[j].append(reward_list[j][k-1][0])
                        else:
                            agent_set[j].append(reward_list[j][k][0])
                        break

    min = 100000
    for i in range(env_width):
        if len(agent_set[i]) < min:
            min = len(agent_set[i])

    print(agent_set)
    print(min)
    vols = calculate_hypervolume(agent_set, reference_point, min)
    print(vols)

    agent_set = [[] for _ in range(env_width)]
    policies = []
    reward_list = []
    total_steps_2 = []

    for i in range(env_width):
        policies.append(Thread(target=train_mo_dqn_agent, args=(env_width, i, reward_list, extended_config, False, True)))
        policies[i].start()

    for i in range(env_width):
        policies[i].join()

    min = 100000
    min_index = -1
    for i in range(env_width):
        print(reward_list[i])
        if len(reward_list[i]) < min:
            min = len(reward_list[i])
            min_index = i

    for i in range(len(reward_list[min_index])):
        training_step = reward_list[min_index][i][1]
        agent_set[min_index].append(reward_list[min_index][i][0])
        total_steps_2.append(reward_list[min_index][i][1])
        for j in range(env_width):
            if j != min_index:
                for k in range(len(reward_list[j])):
                    if reward_list[j][k][1] > training_step:
                        if k > 0:
                            agent_set[j].append(reward_list[j][k-1][0])
                        else:
                            agent_set[j].append(reward_list[j][k][0])
                        break

    min = 100000
    for i in range(env_width):
        if len(agent_set[i]) < min:
            min = len(agent_set[i])

    print("TLO agent set:", agent_set)
    vols_2 = calculate_hypervolume(agent_set, reference_point, min)
    print(vols_2)

    for j in range(len(vols)):
        if envs[0] == 3:
            stat[0].append([vols[j], total_steps[j]])
            stat[1].append([494.5, total_steps[j]])
        elif envs[0] == 5:
            stat[0].append([vols[j], total_steps[j]])
            stat[1].append([1251.67, total_steps[j]])
        else:
            stat[0].append([vols[j], total_steps[j]])
            stat[1].append([1851.67, total_steps[j]])

    for j in range(len(vols_2)):
        if envs[0] == 3:
            stat[2].append([vols_2[j], total_steps_2[j]])
        elif envs[0] == 5:
            stat[2].append([vols_2[j], total_steps_2[j]])
        else:
            stat[2].append([vols_2[j], total_steps_2[j]])

    if envs[0] == 3:
        with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_3/data_hypervolume.bin', 'wb') as file:
            pickle.dump(stat, file)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    elif envs[0] == 5:
        with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_5/data_hypervolume.bin', 'wb') as file:
            pickle.dump(stat, file)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO MO-DQN', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Pareto', linewidth=1)
    else:
        with open('./train/MORL/deep_sea_treasure/hypervolume/parallel/env_7/data_hypervolume.bin', 'wb') as file:
            pickle.dump(stat, file)
        plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear DQN (Env=7)', linewidth=1)
        plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO DQN (Env=7)', linewidth=1)
        plt.plot([x[1] for x in stat[1]], [x[0] for x in stat[1]], 'g--', label='Hypervolume (Env=7)', linewidth=1)

    plt.savefig(
        "./test/MORL/deep_sea_treasure/hypervolume/parallel/deepseatreasure_dqn_performance_test_env_" + str(
            envs[0]) + ".pdf")

    plt.legend(loc='best')
    plt.show()


def mo_dqn_multiple_policy_test_hypervolume_mountain_car(extended_config=True, env_width=6):
    envs = [env_width]
    stat = [[] for _ in range(3)]
    total_steps = []

    reference_point = [-110, -60, -60]

    policies = []
    reward_list = []

    # Run the training process for each pareto solution
    agent_set = [[] for _ in range(env_width)]

    for i in range(env_width):
        policies.append(Thread(target=train_mo_dqn_agent_mountain_car, args=(i, reward_list, extended_config, True, True, True)))
        policies[i].start()

    for i in range(env_width):
        policies[i].join()

    min = 100000
    min_index = -1
    for i in range(env_width):
        print(reward_list[i])
        if len(reward_list[i]) < min:
            min = len(reward_list[i])
            min_index = i

    for i in range(len(reward_list[min_index])):
        training_step = reward_list[min_index][i][1]
        agent_set[min_index].append(reward_list[min_index][i][0])
        total_steps.append(reward_list[min_index][i][1])
        for j in range(env_width):
            if j != min_index:
                for k in range(len(reward_list[j])):
                    if reward_list[j][k][1] >= training_step:
                        if k > 0:
                            agent_set[j].append(reward_list[j][k-1][0])
                        else:
                            agent_set[j].append(reward_list[j][k][0])
                        break

    min = 100000
    for i in range(env_width):
        if len(agent_set[i]) < min:
            min = len(agent_set[i])

    print(agent_set)
    print(min)
    vols = []
    for i in range(min):
        test = []
        for j in range(env_width):
            test.append(agent_set[j][i])
        vol = HVCalculator.get_volume_from_array(test, reference_point)
        vols.append(vol)
    print(vols)

    agent_set = [[] for _ in range(env_width)]
    policies = []
    reward_list = []
    total_steps_2 = []

    for i in range(env_width):
        policies.append(Thread(target=train_mo_dqn_agent_mountain_car, args=(i, reward_list, extended_config, True, False, True)))
        policies[i].start()

    for i in range(env_width):
        policies[i].join()

    min = 100000
    min_index = -1
    for i in range(env_width):
        print(reward_list[i])
        if len(reward_list[i]) < min:
            min = len(reward_list[i])
            min_index = i

    for i in range(len(reward_list[min_index])):
        training_step = reward_list[min_index][i][1]
        agent_set[min_index].append(reward_list[min_index][i][0])
        total_steps_2.append(reward_list[min_index][i][1])
        for j in range(env_width):
            if j != min_index:
                for k in range(len(reward_list[j])):
                    if reward_list[j][k][1] > training_step:
                        if k > 0:
                            agent_set[j].append(reward_list[j][k-1][0])
                        else:
                            agent_set[j].append(reward_list[j][k][0])
                        break

    min = 100000
    for i in range(env_width):
        if len(agent_set[i]) < min:
            min = len(agent_set[i])

    print("TLO agent set:", agent_set)
    print(agent_set)
    print(min)
    vols_2 = []
    for i in range(min):
        test = []
        for j in range(env_width):
            test.append(agent_set[j][i])
        vol = HVCalculator.get_volume_from_array(test, reference_point)
        vols_2.append(vol)
    print(vols_2)

    for j in range(len(vols)):
        stat[0].append([vols[j], total_steps[j]])

    for j in range(len(vols_2)):
        stat[2].append([vols_2[j], total_steps_2[j]])

    with open('./train/MORL/mountain_car/parallel/data_hypervolume2.bin', 'wb') as file:
        pickle.dump(stat, file)
    plt.plot([x[1] for x in stat[0]], [x[0] for x in stat[0]], 'r.-', label='Linear MO-DQN', linewidth=1)
    plt.plot([x[1] for x in stat[2]], [x[0] for x in stat[2]], 'bx-', label='TLO MO-DQN', linewidth=1)

    plt.savefig(
        "./test/MORL/mountain_car/parallel/mountain_car_dqn_performance_test_env2_" + str(
            envs[0]) + ".pdf")

    plt.legend(loc='best')
    plt.show()


def mo_dqn_multi_policy_test(env_size=5):
    policies = []
    reward_list = []
    stat = []

    game = DeepSeaTreasure(width=env_size, seed=100, max_treasure=100)

    # Put game into Fruit wrapper
    env = FruitEnvironment(game)

    # Get pareto set of this game
    pareto_set = env.get_pareto_solutions()

    # Run the training process for each pareto solution
    agent_set = [[0, 0] for x in range(env_size)]

    for i in range(env_size):
        policies.append(Thread(target=train_mo_dqn_agent, args=(env_size, i, reward_list)))
        print(policies[i].start())

    for i in range(env_size):
        policies[i].join()

    for j in range(len(reward_list[0])):
        for k in range(env_size):
            if j < len(reward_list[k]):
                agent_set[k] = reward_list[k][j][0]

        print(agent_set)
        score = additive_epsilon_history(pareto_set, agent_set, env_size)
        stat.append([score, reward_list[0][j][1]])

    if env_size == 3:
        plt.plot([x[1] for x in reward_list[0]], [x[0][0] for x in reward_list[0]], 'r.',
                 [x[1] for x in reward_list[0]], [x[0][1] for x in reward_list[0]], 'c.',
                 [x[1] for x in reward_list[1]], [x[0][0] for x in reward_list[1]], 'g.',
                 [x[1] for x in reward_list[1]], [x[0][1] for x in reward_list[1]], 'm.',
                 [x[1] for x in reward_list[2]], [x[0][0] for x in reward_list[2]], 'b.',
                 [x[1] for x in reward_list[2]], [x[0][1] for x in reward_list[2]], 'y.'
                 )
        plt.savefig("./test/MORL/deepseatreasure_dqn_performance_test_multi_env_3_reward_dist.png")
        plt.show()

        plt.plot([x[1] for x in stat], [x[0] for x in stat], 'bx-', linewidth=1)
        plt.savefig("./test/MORL/deepseatreasure_dqn_performance_test_multi_env_3_eps_history.png")
        plt.show()
    elif env_size == 5:
        plt.plot([x[1] for x in reward_list[0]], [x[0][0] for x in reward_list[0]], 'r.',
                 [x[1] for x in reward_list[0]], [x[0][1] for x in reward_list[0]], 'c.',
                 [x[1] for x in reward_list[1]], [x[0][0] for x in reward_list[1]], 'g.',
                 [x[1] for x in reward_list[1]], [x[0][1] for x in reward_list[1]], 'm.',
                 [x[1] for x in reward_list[2]], [x[0][0] for x in reward_list[2]], 'b.',
                 [x[1] for x in reward_list[2]], [x[0][1] for x in reward_list[2]], 'y.',
                 [x[1] for x in reward_list[3]], [x[0][0] for x in reward_list[3]], 'k.',
                 [x[1] for x in reward_list[3]], [x[0][1] for x in reward_list[3]], 'violet.',
                 [x[1] for x in reward_list[4]], [x[0][0] for x in reward_list[4]], 'purple.',
                 [x[1] for x in reward_list[4]], [x[0][1] for x in reward_list[4]], 'black.',
                 )
        plt.savefig("./test/MORL/deepseatreasure_dqn_performance_test_multi_env_5_reward_dist.png")
        plt.show()

        plt.plot([x[1] for x in stat], [x[0] for x in stat], 'bx-', linewidth=1)
        plt.savefig("./test/MORL/deepseatreasure_dqn_performance_test_multi_env_5_eps_history.png")
        plt.show()
    else:
        plt.plot([x[1] for x in reward_list[0]], [x[0][0] for x in reward_list[0]], 'r.',
                 [x[1] for x in reward_list[0]], [x[0][1] for x in reward_list[0]], 'c.',
                 [x[1] for x in reward_list[1]], [x[0][0] for x in reward_list[1]], 'g.',
                 [x[1] for x in reward_list[1]], [x[0][1] for x in reward_list[1]], 'm.',
                 [x[1] for x in reward_list[2]], [x[0][0] for x in reward_list[2]], 'b.',
                 [x[1] for x in reward_list[2]], [x[0][1] for x in reward_list[2]], 'y.'
                 )
        plt.savefig("./test/MORL/deepseatreasure_dqn_performance_test_multi_env_3_reward_dist.png")
        plt.show()

        plt.plot([x[1] for x in stat], [x[0] for x in stat], 'bx-', linewidth=1)
        plt.savefig("./test/MORL/deepseatreasure_dqn_performance_test_multi_env_3_eps_history.png")
        plt.show()


if __name__ == '__main__':

    # train_multi_objective_agent(3)

    # test_multi_objective_agent()

    # train_multi_objective_agent(3)

    # train_multi_objective_agent_mountain_car()

    # multi_objective_agent_mountain_car_test()

    # mo_ql_test()

    # train_mo_dqn_agent_mountain_car(config=0, extended_config=True, using_cnn=True)

    # mo_dqn_single_policy_test(extended_config=True, env_width=3)

    # mo_dqn_multi_policy_test()

    # game = DeepSeaTreasure(width=7, seed=100, max_steps=100, max_treasure=100)
    #
    # print(game.pareto_solutions)

    # train_mo_dqn_agent(env_size=7, config=4, extended_config=True)

    # -117 -52 -61

    #game = DeepSeaTreasure(graphical_state=False, width=5, seed=100, render=True,
    #                       max_treasure=100)

    #print(game.treasures, game.pareto_solutions)

    # print(calculate_hypervolume([[x] for x in game.pareto_solutions], [-20, -10], 1))
    #
    # print(calculate_hypervolume([[[1, -3]], [[100, -7]], [[26.25, -5]]], [-20, -10], 1))

    # mo_dqn_single_policy_test_hypervolume(env_width=5)

    # get_data(env_width=3, parallel=False, times=8)

    # get_data_combined(env_width=5)

    # mo_dqn_multiple_policy_test_hypervolume(env_width=5)

    # mo_dqn_single_policy_test_hypervolume_mountain_car()

    # mo_dqn_multiple_policy_test_hypervolume_mountain_car()

    # get_data_mc(parallel=False, times=10, skip=1)

    get_data_mc_combined()

    # train_mo_dqn_agent_mountain_car(0, None, True, True, False, False)

    #train_mo_dqn_agent_mountain_car(config=0, reward_list=None, extended_config=True, using_cnn=True,
    #                                is_linear=False, parallel=False)
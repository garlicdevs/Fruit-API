import numpy as np
from fruit.envs.games.tic_tac_toe.engine import TicTacToe
import pickle


# This program is a supplementary material for the book: "Reinforcement Learning: An Introduction - Sutton and Barto"
# Train a Tic Tac Toe agent with a random agent using Temporal Difference learning

# Value tables
values_table = {'0': .5}


# Parameters
initial_step_size_parameter = 0.9
final_step_size_parameter = 0
training_steps = 100000
initial_epsilon = 0.9
final_epsilon = 0


def get_greedy_action(current_state, possible_actions, current_epsilon, debug=False):
    if debug:
        print('Current state', current_state, current_epsilon)
        print('Possible actions', possible_actions)

    if np.random.uniform(0, 1) < current_epsilon:
        return np.random.choice(possible_actions)
    else:
        probs = []
        for a in possible_actions:
            next_values = []
            next_state = int(current_state + (3 ** a))
            if str(next_state) in values_table:
                return a
            for b in possible_actions:
                if a != b:
                    next_state_ = str(int(next_state + 2 * (3 ** b)))
                    if next_state_ in values_table:
                        next_values.append(values_table[next_state_])
            if len(next_values) == 0:
                min_val = -1
            else:
                min_val = max(next_values)
            probs.append(min_val)

        if debug:
            print('Possible states', probs)

        if max(probs) == -1:
            return np.random.choice(possible_actions)
        else:
            return possible_actions[np.argmax(probs)]


def update_value_table(old_state, new_state, reward, step_param):
    o_state_ = str(old_state)
    n_state_ = str(new_state)

    old_V = values_table[o_state_]

    if n_state_ in values_table:
        new_V = values_table[n_state_]
    else:
        if reward > 0:
            values_table[n_state_] = 1
        elif reward < 0:
            values_table[n_state_] = 0
        else:
            values_table[n_state_] = 0.5
        new_V = values_table[n_state_]

    old_V = old_V + step_param * (new_V - old_V)
    values_table[o_state_] = old_V


def train_with_random_enemy(output_model):

    current_step_size_param = initial_step_size_parameter
    current_epsilon = initial_epsilon

    # Prepare game engine
    game = TicTacToe()
    state = game.reset()
    print('Reset state', state)
    player_state = state
    num_actions = game.get_num_of_actions()
    print('Num of Actions', num_actions)

    # Training loop
    for i in range(training_steps):
        # Update current step-size param
        current_step_size_param -= (initial_step_size_parameter - final_step_size_parameter)/training_steps
        if current_step_size_param < final_step_size_parameter:
            current_step_size_param = final_step_size_parameter
        print('Current step-size param', current_step_size_param)

        # Update current epsilon param
        current_epsilon -= (initial_epsilon - final_epsilon)/training_steps
        if current_epsilon < final_epsilon:
            current_epsilon = final_epsilon

        # Player
        print('------ PLAYER PHASE -----')
        actions = game.get_possible_actions()
        greedy_action = get_greedy_action(player_state, actions, current_epsilon)
        print('Action', greedy_action)
        player_reward = game.step(greedy_action)
        print('Reward', player_reward)
        player_state_next = game.get_state()
        print('State', player_state_next)
        terminal = game.is_terminal()
        game.print()
        if terminal:
            update_value_table(player_state, player_state_next, player_reward, current_step_size_param)
            player_state = game.reset()
            continue
        print('------ ----------- -----\n')

        # Enemy
        print('------ ENEMY PHASE -----')
        actions = game.get_possible_actions()
        rand_action = np.random.choice(actions)
        print('Action', rand_action)
        enemy_reward = game.step(rand_action, is_enemy=True)
        print('Reward', enemy_reward)
        enemy_state_next = game.get_state()
        print('State', enemy_state_next)
        update_value_table(player_state, enemy_state_next, -enemy_reward, current_step_size_param)
        player_state = enemy_state_next
        terminal = game.is_terminal()
        game.print()
        print('------ ----------- -----\n')
        if terminal:
            player_state = game.reset()

    print(values_table)
    with open(output_model, 'wb+') as file:
        pickle.dump(values_table, file)


def evaluate(input_model):
    with open(input_model, 'rb') as file:
        global values_table
        values_table = pickle.load(file)
        print('Values table', len(values_table), values_table)

    # Prepare game engine
    game = TicTacToe()
    state = game.reset()
    print('Reset state', state)
    player_state = state
    num_actions = game.get_num_of_actions()
    print('Num of Actions', num_actions)

    # Training loop
    for i in range(100):
        # Player
        print('------ PLAYER PHASE -----')
        actions = game.get_possible_actions()
        greedy_action = get_greedy_action(player_state, actions, 0, debug=True)
        print('Action', greedy_action)
        player_reward = game.step(greedy_action)
        print('Reward', player_reward)
        player_state_next = game.get_state()
        print('State', player_state_next)
        terminal = game.is_terminal()
        game.print()
        if terminal:
            break
        print('------ ----------- -----\n')

        # Enemy
        print('------ ENEMY PHASE -----')
        actions = game.get_possible_actions()
        rand_action = np.random.choice(actions)
        print('Action', rand_action)
        enemy_reward = game.step(rand_action, is_enemy=True)
        print('Reward', enemy_reward)
        enemy_state_next = game.get_state()
        print('State', enemy_state_next)
        terminal = game.is_terminal()
        player_state = enemy_state_next
        game.print()
        print('------ ----------- -----\n')
        if terminal:
            break


if __name__ == '__main__':
    model = 'c1_model.b'

    # train_with_random_enemy(output_model=model)

    evaluate(input_model=model)

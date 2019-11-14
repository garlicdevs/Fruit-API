import random

from fruit.monitor.monitor import AgentMonitor
from tensorforce import util

from fruit.learners.base import Learner
from tensorforce.agents import Agent

from fruit.envs.ale import ALEEnvironment
from fruit.envs.base import BaseEnvironment
from fruit.envs.gym import GymEnvironment
from fruit.plugins.base import Plugin
from tensorforce.environments import Environment, ArcadeLearningEnvironment, OpenAIGym, MazeExplorer, OpenSim, \
    OpenAIRetro
import numpy as np


# The code is written based on TensorForce source code
class TFEnvironment(Environment):
    def __init__(self, fruit_environment, **kwargs):
        super().__init__()
        if isinstance(fruit_environment, BaseEnvironment):
            self.environment = fruit_environment
            if self.environment.get_processor() is not None:
                raise ValueError('Do not use state processor with TensorForce !')
        else:
            raise ValueError('Environment must be from FruitAPI !')

        self.__max_episode_timesteps = False

        state_space = self.environment.get_state_space()
        self.states_spec = dict(type='float', shape=tuple(state_space.get_shape()))

        action_ranges, _ = self.environment.get_action_space().get_range()
        self.actions_spec = dict(type='int', num_values=len(action_ranges))

        self.__timesteps = 0

        if isinstance(fruit_environment, GymEnvironment):
            self.__max_episode_timesteps = None
            _, self.__max_episode_timesteps = OpenAIGym.create_level(level=self.environment.env_name,
                                                                     max_episode_timesteps=self.__max_episode_timesteps,
                                                                     reward_threshold=None, tags=None, **kwargs)

            self.states_spec = OpenAIGym.specs_from_gym_space(
                space=self.environment.env.observation_space, ignore_value_bounds=True)

            self.actions_spec = OpenAIGym.specs_from_gym_space(
                space=self.environment.env.action_space, ignore_value_bounds=False)

    def __str__(self):
        return super().__str__()

    def states(self):
        return self.states_spec

    def actions(self):
        return self.actions_spec

    def close(self):
        self.reset()

    def get_states(self):
        state = self.environment.get_state()
        if isinstance(self.environment, GymEnvironment):
            state = OpenAIGym.flatten_state(state=state, states_spec=self.states_spec)
        else:
            state = state.astype(dtype=np.float32) / 255.0
        return state

    def reset(self):
        self.__timesteps = 0
        self.environment.reset()
        return self.get_states()

    def execute(self, actions):
        if isinstance(self.environment, GymEnvironment):
            actions = OpenAIGym.unflatten_action(action=actions)

        reward = self.environment.step(actions)
        terminal = self.environment.is_terminal()
        self.__timesteps += 1
        if self.__max_episode_timesteps is not None:
            if self.__timesteps > self.__max_episode_timesteps:
                terminal = 2
            elif terminal:
                terminal = 1
            else:
                terminal = 0

        states = self.get_states()

        return states, terminal, reward

    def max_episode_timesteps(self):
        if self.__max_episode_timesteps is False:
            return super().max_episode_timesteps()
        else:
            return self.__max_episode_timesteps


class FTEnvironment(BaseEnvironment):
    def __init__(self, tf_environment, state_processor=None):
        self.environment = tf_environment
        self.current_state = None
        self.current_reward = 0
        self.current_terminal = False
        self.current_steps = 0
        self.processor = state_processor

    def clone(self):
        return FTEnvironment(self.environment, state_processor=self.processor)

    def step(self, actions):
        self.current_state, self.current_terminal, self.current_reward = self.environment.execute(actions)
        if self.processor is not None:
            self.current_state = self.processor.process(self.current_state)
        self.current_steps += 1
        return self.current_reward

    def reset(self):
        self.current_steps = 0
        return self.environment.reset()

    def get_current_steps(self):
        return self.current_steps

    def get_action_space(self, act_space=None):
        from fruit.types.priv import Space
        if act_space is None:
            act_space = self.environment.actions()
        print(act_space)
        if 'num_values' in act_space:
            return Space(0, act_space['num_values'] - 1, True)
        elif 'num_actions' in act_space:
            return Space(0, act_space['num_actions']-1, True)
        elif 'shape' in act_space:
            if 'min_value' in act_space and 'max_value' in act_space:
                min_value = act_space['min_value']
                max_value = act_space['max_value']
                if np.asarray(act_space['min_value']).shape != act_space['shape']:
                    min_value = np.full(act_space['shape'], act_space['min_value'])
                    max_value = np.full(act_space['shape'], act_space['max_value'])
            elif 'type' in act_space:
                if isinstance(act_space['type'], bool) or act_space['type'] == 'bool':
                    min_value = np.full(act_space['shape'], 0)
                    max_value = np.full(act_space['shape'], 1)
                    return Space(min_value, max_value, True)
                else:
                    min_value = np.full(act_space['shape'], 0)
                    max_value = np.full(act_space['shape'], 1)
            else:
                min_value = np.full(act_space['shape'], 0)
                max_value = np.full(act_space['shape'], 1)

            return Space(min_value, max_value, False)
        elif 'gymtpl0' in act_space:
            values = []
            for i in range(len(act_space)):
                key = 'gymtpl' + str(i)
                values.append(self.get_action_space(act_space[key]))
            return tuple(values)
        else:
            raise ValueError('Action space {} is not supported !'.format(act_space))

    def get_state_space(self, st_space=None):
        from fruit.types.priv import Space
        if st_space is None:
            st_space = self.environment.states()
        print(st_space)
        if 'num_values' in st_space:
            return Space(0, st_space['num_values'] - 1, True)
        elif 'shape' in st_space:
            if self.processor is None:
                shape = st_space['shape']
            else:
                shape = self.current_state.shape
            min_value = np.zeros(shape)
            max_value = np.full(shape, 1.)
            return Space(min_value, max_value, True)
        elif 'gymtpl0' in st_space:
            values = []
            for i in range(len(st_space)):
                key = 'gymtpl' + str(i)
                values.append(self.get_state_space(st_space[key]))
            return tuple(values)
        else:
            raise ValueError('State space {} is not supported !'.format(st_space))

    def step_all(self, action):
        self.current_state, self.current_terminal, self.current_reward = self.environment.execute(action)
        if self.processor is not None:
            self.current_state = self.processor.process(self.current_state)
        self.current_steps += 1
        return self.current_state, self.current_reward, self.current_terminal, None

    def get_state(self):
        return self.current_state

    def is_terminal(self):
        return self.current_terminal

    def is_atari(self):
        if isinstance(self.environment, ArcadeLearningEnvironment):
            return True
        else:
            return False

    def is_render(self):
        raise NotImplementedError

    def get_number_of_objectives(self):
        return 1

    def get_number_of_agents(self):
        return 1

    def get_processor(self):
        return self.processor


class TensorForceLearner(Learner):
    def __init__(self, agent, learner, environment, p_network, global_dict, report_frequency,
                 algorithm, callback=None, callback_episode_frequency=None, callback_timestep_frequency=None,
                 parallel_interactions=1, num_episodes=None, **kwargs
                 ):
        if isinstance(environment, BaseEnvironment):
            fruit_environment = environment
            self.tf_environment = TensorForcePlugin.convert(environment)
        else:
            environment = Environment.create(environment=environment)
            fruit_environment = TensorForcePlugin.convert(environment)
            self.tf_environment = environment

        super().__init__(agent=agent, name=learner, environment=fruit_environment, network=p_network,
                         global_dict=global_dict,
                         report_frequency=report_frequency)
        self.algorithm = algorithm
        self.tf_agent = Agent.create(
            algorithm, self.tf_environment, **kwargs
        )
        if not self.tf_agent.model.is_initialized:
            self.tf_agent.initialize()

        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.episode_seconds = list()

        self.parallel_interactions = parallel_interactions
        if num_episodes is None:
            self.num_episodes = float('inf')
        else:
            self.num_episodes = num_episodes

        assert callback_episode_frequency is None or callback_timestep_frequency is None
        if callback_episode_frequency is None and callback_timestep_frequency is None:
            callback_episode_frequency = 1
        if callback_episode_frequency is None:
            self.callback_episode_frequency = float('inf')
        else:
            self.callback_episode_frequency = callback_episode_frequency
        if callback_timestep_frequency is None:
            self.callback_timestep_frequency = float('inf')
        else:
            self.callback_timestep_frequency = callback_timestep_frequency
        if callback is None:
            self.callback = (lambda r: True)
        elif util.is_iterable(x=callback):
            def sequential_callback(runner):
                result = True
                for fn in callback:
                    x = fn(runner)
                    if isinstance(result, bool):
                        result = result and x
                return result

            self.callback = sequential_callback
        else:
            def boolean_callback(runner):
                result = callback(runner)
                if isinstance(result, bool):
                    return result
                else:
                    return True

            self.callback = boolean_callback

    def __del__(self):
        if isinstance(self.tf_agent, Agent):
            self.tf_agent.close()
        if isinstance(self.tf_environment, Environment):
            self.tf_environment.close()

    @staticmethod
    def get_default_number_of_learners():
        return 1

    def report(self, reward, mean_horizon=10):
        mean_reward = float(np.mean(self.episode_rewards[-mean_horizon:]))
        mean_ts_per_ep = int(np.mean(self.episode_timesteps[-mean_horizon:]))
        mean_sec_per_ep = float(np.mean(self.episode_seconds[-mean_horizon:]))

        print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Episode steps:',
              self.environment.get_current_steps(), 'Total steps:', self.global_dict[AgentMonitor.Q_GLOBAL_STEPS],
              'Mean reward:', mean_reward, 'Mean steps/episode:', mean_ts_per_ep, 'Mean seconds/episode:',
              mean_sec_per_ep)

    def reset(self):
        super().reset()
        self.tf_agent.reset()

    def episode_end(self, episode_reward, episode_steps, episode_secs):
        self.episode_rewards.append(episode_reward)
        self.episode_timesteps.append(episode_steps)
        self.episode_seconds.append(episode_secs)

        if self.eps_count % self.callback_episode_frequency == 0 and not self.callback(self):
            return

        if self.num_episodes != float('inf') and self.eps_count != 0 and self.eps_count % self.num_episodes == 0:
            self.global_dict[AgentMonitor.Q_FINISH] = True

    def get_action(self, state):
        return self.tf_agent.act(states=state)

    def save_model(self, str):
        print('Not implemented yet !')

    def update(self, state, action, reward, next_state, terminal):
        # if self.history_length > 1:
        #     self.frame_buffer.add_state(state)
        #
        # if not self.testing:
        #     if self.history_length > 1:
        #         current_s = self.frame_buffer.get_buffer()[0]
        #         next_s = self.frame_buffer.get_buffer_add_state(next_state)[0]
        #     else:
        #         current_s = state
        #         next_s = next_state
        #     self.data_dict['states'].append(current_s)
        #     self.data_dict['actions'].append(action)
        #     self.data_dict['rewards'].append(reward)
        #     self.data_dict['next_states'].append(next_s)
        #     self.data_dict['terminals'].append(terminal)

        self.step_count += 1
        self.global_dict[AgentMonitor.Q_GLOBAL_STEPS] += 1

        if not self.testing:
            if self.step_count % self.callback_timestep_frequency == 0:
                self.callback(self)

            self.tf_agent.observe(terminal=terminal, reward=reward)

            # if self.step_count % self.async_update_steps == 0 or terminal:
            #     logging = self.global_dict[AgentMonitor.Q_LOGGING]
            #     self.current_learning_rate = self.learning_rate_annealer.anneal(
            #         self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])
            #     self.data_dict['learning_rate'] = self.current_learning_rate
            #     self.global_dict[AgentMonitor.Q_LEARNING_RATE] = self.current_learning_rate
            #     if logging:
            #         self.global_dict[AgentMonitor.Q_LOGGING] = False
            #         self.data_dict['logging'] = True
            #         summary = self.network.train_network(self.data_dict)
            #         self.global_dict[AgentMonitor.Q_WRITER].\
            #             add_summary(summary, global_step=self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])
            #     else:
            #         self.data_dict['logging'] = False
            #         self.network.train_network(self.data_dict)
            #     self.reset_batch()


class TensorForcePlugin(Plugin):
    @staticmethod
    def convert(environment):
        if isinstance(environment, BaseEnvironment):
            return TFEnvironment(environment)
        elif isinstance(environment, Environment):
            return FTEnvironment(environment)
        else:
            raise ValueError('Environment is not supported !')

    def get_config(self):
        pass

    def get_learner(self):
        return TensorForceLearner


def compatible_1():
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    fruit_env = ALEEnvironment(rom_name=ALEEnvironment.BREAKOUT,
                               state_processor=None)
    state = fruit_env.get_state_space()
    print(state.get_range())
    print(tuple(state.get_shape()))
    print(fruit_env.get_action_space().get_range())
    print(fruit_env.reset())
    print(fruit_env.get_state())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    env = ArcadeLearningEnvironment('../envs/roms/breakout.bin')
    state = env.states()
    print(state)
    print(env.actions())
    print(env.reset())
    print(env.get_states())
    print(env.execute(0))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    env = TFEnvironment(fruit_environment=fruit_env)
    print(env.states())
    print(env.actions())
    print(env.get_states())
    print(env.execute(0))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')


def compatible_2():
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    fruit_env = GymEnvironment(env_name='CartPole-v1')
    state = fruit_env.get_state_space()
    print(state.get_range())
    print(tuple(state.get_shape()))
    print(fruit_env.get_action_space().get_range())
    print(fruit_env.reset())
    print(fruit_env.get_state())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    env = OpenAIGym(level='CartPole-v1')
    state = env.states()
    print(state)
    print(env.actions())
    print(env.reset())
    print(env.execute(0))
    print(env.max_episode_timesteps())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    env = TFEnvironment(fruit_environment=fruit_env)
    print(env.states())
    print(env.actions())
    print(env.get_states())
    print(env.execute(0))
    print(env.max_episode_timesteps())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')


def get_random_action(is_discrete, action_range, action_space):
    if is_discrete:
        if len(action_range) == 2 and isinstance(action_range[0], (list, np.ndarray, tuple)):
            action = [random.randint(action_range[0][i], action_range[1][i]) for i in range(len(action_range[0]))]
        else:
            action = random.randint(0, len(action_range) - 1)
    else:
        rand = np.random.rand(*tuple(action_space.get_shape()))
        action = np.multiply(action_range[1] - action_range[0], rand) + action_range[0]
    return action


def create_random_agent(env):
    fruit_env = TensorForcePlugin.convert(environment=env)

    state_space = fruit_env.get_state_space()
    if isinstance(state_space, tuple):
        for s in state_space:
            print(s.get_range(), s.get_shape())
    else:
        print(state_space.get_range(), state_space.get_shape())

    action_space = fruit_env.get_action_space()
    is_discrete = False
    action_range = None
    if isinstance(action_space, tuple):
        for s in action_space:
            action_range, is_discrete = s.get_range()
            print(action_range, s.get_shape())
    else:
        action_range, is_discrete = action_space.get_range()
        print(action_range, action_space.get_shape())

    fruit_env.reset()
    for i in range(1000):
        if isinstance(action_space, tuple):
            action = []
            for s in action_space:
                action_range, is_discrete = s.get_range()
                action.append(get_random_action(is_discrete, action_range, s))
        else:
            action = get_random_action(is_discrete, action_range, action_space)
        print(action)
        reward = fruit_env.step(action)
        next_state = fruit_env.get_state()
        state = next_state
        terminal = fruit_env.is_terminal()
        print(state, action, reward, terminal)

        if terminal:
            fruit_env.reset()
            break


def conversion_1():
    env = ArcadeLearningEnvironment(level='../envs/roms/breakout.bin', visualize=True)
    create_random_agent(env)


def conversion_2():
    env = OpenAIGym(level='FrozenLake8x8-v0', visualize=False)
    create_random_agent(env)


def conversion_3():
    env = MazeExplorer(level=1, visualize=True)
    create_random_agent(env)


def conversion_4():
    env = OpenSim(level='Arm2D', visualize=True)
    create_random_agent(env)


def conversion_5():
    print(OpenAIRetro.levels())
    env = OpenAIRetro('Airstriker-Genesis', visualize=True)
    create_random_agent(env)


if __name__ == '__main__':
    conversion_5()

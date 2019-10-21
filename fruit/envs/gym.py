from fruit.envs.base import BaseEnvironment
from fruit.state.processor import AtariProcessor
from fruit.types.priv import Space
import gym
import numpy as np


class GymEnvironment(BaseEnvironment):
    def __init__(self, env_name, state_processor=AtariProcessor()):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.atari = GymEnvironment._is_atari(env_name)
        self.processor = state_processor
        self.cur_steps = 0
        self.current_state = None
        self.terminal = False

    @staticmethod
    def _is_atari(env_name):
        for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
                     'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
                     'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
                     'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
                     'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
                     'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
                     'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
                     'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
                     'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:

            name = ''.join([g.capitalize() for g in game.split('_')])
            if name in env_name:
                return True
        return False

    def is_atari(self):
        return self.atari

    def clone(self):
        return GymEnvironment(self.env_name, self.processor)

    def get_state(self):
        return self.current_state

    def reset(self):
        self.cur_steps = 0
        self.terminal = False
        if self.processor is not None:
            self.current_state = self.processor.process(self.env.reset())
        else:
            self.current_state = self.env.reset()
        return self.current_state

    def is_terminal(self):
        return self.terminal

    def step(self, action):
        self.cur_steps += 1
        observation, reward, terminal, _ = self.env.step(action)
        if self.processor is not None:
            self.current_state = self.processor.process(observation)
        else:
            self.current_state = observation
        self.terminal = terminal
        return reward

    def step_all(self, action):
        self.cur_steps += 1
        observation, reward, terminal, info = self.env.step(action)
        if self.processor is not None:
            self.current_state = self.processor.process(observation)
        else:
            self.current_state = observation
        self.terminal = terminal
        return self.current_state, reward, terminal, info

    def get_state_space(self):
        if self.processor is None:
            return Space.convert_openai_space(self.env.observation_space)
        else:
            self.reset()
            shape = self.current_state.shape
            min = np.zeros(shape, dtype=np.uint8)
            max = np.full(shape, 255, dtype=np.uint8)
            return Space(min, max, True)

    def render(self):
        self.env.render()

    def get_action_space(self):
        return Space.convert_openai_space(self.env.action_space)

    def set_seed(self, seed):
        self.env.seed(seed)

    def get_current_steps(self):
        return self.cur_steps

    def get_number_of_objectives(self):
        return 1

    def get_number_of_agents(self):
        return 1

import random
import deepmind_lab
from fruit.envs.base import BaseEnvironment
import numpy as np


class DeepMindLabEnvironment(BaseEnvironment):
    def __init__(self, screen_width=300, screen_height=300, runfiles_path='', state_processor=None,
                 level_script='tests/empty_room_test', frame_skip=1, seed=None, level_directory=''):
        self.width = screen_width
        self.height = screen_height
        self.runfiles_path = runfiles_path
        self.level_script = level_script
        self.frame_skip = frame_skip
        self.seed = seed
        self.level_directory = level_directory
        self.processor = state_processor
        self.current_state = None
        self.mode = 'RGB_INTERLEAVED'

        config = {
            'width': str(screen_width),
            'height': str(screen_height),
            'levelDirectory': str(level_directory)
        }

        if self.runfiles_path:
            deepmind_lab.set_runfiles_path(self.runfiles_path)
        self.env = deepmind_lab.Lab(level_script, [self.mode], config=config)
        self.number_of_actions = len(self.env.action_spec())

    def clone(self):
        return DeepMindLabEnvironment(screen_width=self.width, screen_height=self.height, state_processor=self.processor,
                                      runfiles_path=self.runfiles_path, level_script=self.level_script,
                                      frame_skip=self.frame_skip, seed=self.seed, level_directory=self.level_directory)

    def get_state(self):
        observation = self.env.observations()[self.mode]
        if self.processor is not None:
            self.current_state = self.processor.process(observation)
        else:
            self.current_state = observation
        return self.current_state

    def reset(self):
        self.env.reset(seed=self.seed)
        return self.get_state()

    def is_terminal(self):
        return not self.env.is_running()

    def step(self, action):
        return self.env.step(action, num_steps=self.frame_skip)

    def step_all(self, action):
        reward = self.step(action)
        self.get_state()
        terminal = self.is_terminal()
        return self.current_state, reward, terminal, None

    def get_state_space(self):
        from fruit.types.priv import Space
        state_spec = self.env.observation_spec()
        for d in state_spec:
            if d['name'] == self.mode:
                shape = d['shape']
                return Space(np.full(shape, 0), np.full(shape, 255), True)
        return None

    def render(self):
        return

    def get_action_space(self):
        from fruit.types.priv import Space
        action_spec = self.env.action_spec()
        min_values = [d['min'] for d in action_spec]
        max_values = [d['max'] for d in action_spec]
        return Space(min_values, max_values, True)

    def get_current_steps(self):
        return self.env.num_steps()

    def get_number_of_objectives(self):
        return 1

    def get_number_of_agents(self):
        return 1

    def get_number_of_actions(self):
        return self.number_of_actions


if __name__ == '__main__':
    from fruit.utils.image import save_rgb_image

    env = DeepMindLabEnvironment()

    num_of_actions = env.get_number_of_actions()
    action_space = env.get_action_space()

    action_choice = random.randint(0, num_of_actions - 1)
    action_amount = random.randint(action_space.get_min()[action_choice],
                                   action_space.get_max()[action_choice])
    action = np.zeros([num_of_actions], dtype=np.intc)
    action[action_choice] = action_amount

    state = env.reset()
    for i in range(100):
        reward = env.step(action)
        next_state = env.get_state()
        is_terminal = env.is_terminal()
        state = next_state

        save_rgb_image(state, './test/step_{}.jpeg'.format(i))

        if is_terminal:
            env.reset()
            break
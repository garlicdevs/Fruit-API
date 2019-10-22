from fruit.envs.base import BaseEnvironment
import os
from os import listdir
from os.path import isfile, join
import os.path
from fruit.state.processor import AtariProcessor, Image
from ale_python_interface import ALEInterface
import numpy as np
from fruit.types.priv import Space


class ALEEnvironment(BaseEnvironment):

    # 63 games
    ADVENTURE = "adventure"
    AIR_RAID = "air_raid"
    ALIEN = "alien"
    AMIDAR = "amidar"
    ASSAULT = "assault"
    ASTERIX = "asterix"
    ASTEROIDS = "asteroids"
    ATLANTIS = "aslantis"
    BANK_HEIST = "bank_heist"
    BATTLE_ZONE = "battle_zone"
    BEAM_RIDER = "beam_rider"
    BERZERK = "berzerk"
    BOWLING = "bowling"
    BOXING = "boxing"
    BREAKOUT = "breakout"
    CARNIVAL = "carnival"
    CENTIPEDE = "centipede"
    CHOPPER_COMMAND = "chopper_command"
    CRAZY_CLIMBER = "crazy_climber"
    DEFENDER = "defender"
    DEMON_ATTACK = "demon_attack"
    DOUBLE_DUNK = "double_dunk"
    ELEVATOR_ACTION = "elevator_action"
    ENDURO = "enduro"
    FISHING_DERBY = "fishing_derby"
    FREEWAY = "freeway"
    FROSTBITE = "frostbite"
    GOPHER = "gopher"
    GRAVITAR = "gravitar"
    HERO = "hero"
    ICE_HOCKEY = "ice_hockey"
    JAMESBOND = "jamesbond"
    JOURNEY_ESCAPE = "journey_escape"
    KABOOM = "kaboom"
    KANGAROO = "kangaroo"
    KRULL = "krull"
    KUNGFU_MASTER = "kung_fu_master"
    MONTEZUMA = "montezuma_revenge"
    MS_PACMAN = "ms_pacman"
    UNKNOWN = "name_this_game"
    PHOENIX = "phoenix"
    PITFALL = "pitfall"
    PONG = "pong"
    POOYAN = "pooyan"
    PRIVATE_EYE = "private_eye"
    QBERT = "qbert"
    RIVERRAID = "riverraid"
    ROAD_RUNNER = "road_runner"
    ROBOTANK = "robotank"
    SEAQUEST = "seaquest"
    SKIING = "skiing"
    SOLARIS = "solaris"
    SPACE_INVADERS = "space_invaders"
    STAR_GUNNER = "star_gunner"
    TENNIS = "tennis"
    TIME_PILOT = "time_pilot"
    TUTANKHAM = "tutankham"
    UP_N_DOWN = "up_n_down"
    VENTURE = "venture"
    VIDEO_PINBALL = "video_pinball"
    WIZARD_OF_WOR = "wizard_of_wor"
    YARS_REVENGE = "yars_revenge"
    ZAXXON = "zaxxon"

    def __init__(self, rom_name, frame_skip=4, repeat_action_probability=0., max_episode_steps=10000,
                 loss_of_life_termination=False, loss_of_life_negative_reward=False,
                 bitwise_max_on_two_consecutive_frames=False, is_render=False, seed=None, startup_policy=None,
                 disable_actions=None, num_of_sub_actions=-1,
                 state_processor=AtariProcessor(resize_shape=(84, 84), convert_to_grayscale=True)):

        os.environ['SDL_VIDEO_CENTERED'] = '1'

        file_exist = isfile(ALEEnvironment.get_rom_path(rom_name))
        if not file_exist:
            raise ValueError("Rom not found ! Please put rom " + rom_name + ".bin into: " +
                             ALEEnvironment.get_rom_path())

        self.__rom_name = rom_name
        self.__ale = ALEInterface()

        if frame_skip < 0:
            print("Invalid frame_skip param ! Set default frame_skip = 4")
            self.__frame_skip = 4
        else:
            self.__frame_skip = frame_skip

        if repeat_action_probability < 0 or repeat_action_probability > 1:
            raise ValueError("Invalid repeat_action_probability")
        else:
            self.__repeat_action_probability = repeat_action_probability

        self.__max_episode_steps = max_episode_steps
        self.__loss_of_life_termination = loss_of_life_termination
        self.__loss_of_life_negative_reward = loss_of_life_negative_reward
        self.__max_2_frames = bitwise_max_on_two_consecutive_frames

        # Max 2 frames only work with grayscale
        self.__grayscale = False
        if state_processor is not None and type(state_processor) is AtariProcessor and state_processor.get_grayscale():
            self.__grayscale = True

        if self.__max_2_frames and self.__frame_skip > 1 and self.__grayscale:
            self.__max_2_frames = True
        else:
            self.__max_2_frames = False

        self.__is_render = is_render
        self.__processor = state_processor

        if seed is None or seed <= 0 or seed >= 9999:
            if seed is not None and (seed < 0 or seed >= 9999):
                print("Invalid seed ! Default seed = randint(0, 9999")
            self.__seed = np.random.randint(0, 9999)
            self.__random_seed = True
        else:
            self.__random_seed = False
            self.__seed = seed

        self.__current_steps = 0
        self.__is_life_lost = False
        self.__is_terminal = False
        self.__current_lives = 0
        self.__action_reduction = num_of_sub_actions
        self.__scr_width, self.__scr_height, self.__action_set = self.__init_ale()
        self.__prev_buffer = np.empty((self.__scr_height, self.__scr_width, 3), dtype=np.uint8)
        self.__current_buffer = np.empty((self.__scr_height, self.__scr_width, 3), dtype=np.uint8)
        self.__current_state = None
        self.__prev_state = None
        self.__startup_policy = startup_policy
        if disable_actions is None:
            self.__dis_act = []
        else:
            self.__dis_act = disable_actions

        if self.__processor.get_number_of_objectives() > 1:
            self.__multi_objs = True
        else:
            self.__multi_objs = False

    def __init_ale(self):

        self.__ale.setBool(b'display_screen', self.__is_render)

        if self.__max_2_frames and self.__frame_skip > 1:
            self.__ale.setInt(b'frame_skip', 1)
        else:
            self.__ale.setInt(b'frame_skip', self.__frame_skip)

        self.__ale.setInt(b'random_seed', self.__seed)
        self.__ale.setFloat(b'repeat_action_probability', self.__repeat_action_probability)
        self.__ale.setBool(b'color_averaging', False)

        self.__ale.loadROM(ALEEnvironment.get_rom_path(self.__rom_name).encode())

        width, height = self.__ale.getScreenDims()
        return width, height, self.__ale.getMinimalActionSet()

    def clone(self):
        if self.__random_seed:
            seed = np.random.randint(0, 9999)
        else:
            seed = self.__seed

        return ALEEnvironment(self.__rom_name, self.__frame_skip, self.__repeat_action_probability,
                              self.__max_episode_steps,
                              self.__loss_of_life_termination, self.__loss_of_life_negative_reward, self.__max_2_frames,
                              self.__is_render, seed, self.__startup_policy,
                              self.__dis_act, self.__action_reduction, self.__processor.clone())

    def step_all(self, a):
        if isinstance(a, (list, np.ndarray)):
            if len(a) <= 0:
                raise ValueError('Empty action list !')
            a = a[0]
        self.__current_steps += 1
        act = self.__action_set[a]
        rew = self._step(act)
        next_state = self.get_state()
        _is_terminal = self.is_terminal()
        return next_state, rew, _is_terminal, self.__current_steps

    def reset(self):
        self.__ale.reset_game()
        self.__current_lives = self.__ale.lives()
        self.__is_life_lost = False
        self.__is_terminal = False
        self.__current_state = None
        self.__prev_state = None

        action_space = self.get_action_space()
        v_range, is_range = action_space.get_range()
        if len(v_range) > 1:
            self.step(1)

        # No op steps
        if self.__startup_policy is not None:
            max_steps = int(self.__startup_policy.get_max_steps())
            for _ in range(max_steps):
                act = self.__startup_policy.step(self.get_state(), action_space)
                self.step(act)

        # Start training from this point
        self.__current_steps = 0

        # Reset processor
        self.__processor.reset()

        return self.get_state()

    def _pre_step(self, act):
        if self.__max_2_frames and self.__frame_skip > 1:
            rew = 0
            for i in range(self.__frame_skip - 2):
                rew += self.__ale.act(act)
                self.__prev_buffer = self.__ale.getScreenRGB(self.__prev_buffer)

            self.__prev_buffer = self.__ale.getScreenRGB(self.__prev_buffer)

            rew += self.__ale.act(act)

            self.__current_buffer = self.__ale.getScreenRGB(self.__current_buffer)

            self.__is_terminal = self.__ale.game_over()

            self.__prev_state = self.__processor.process(self.__prev_buffer)

            self.__current_state = self.__processor.process(self.__current_buffer)

            self.__current_state = np.maximum.reduce([self.__prev_state, self.__current_state])
        else:
            rew = self.__ale.act(act)
            self.__current_buffer = self.__ale.getScreenRGB(self.__current_buffer)
            self.__is_terminal = self.__ale.game_over()

            if self.__processor is not None:
                self.__current_state = self.__processor.process(self.__current_buffer)

        if self.__multi_objs and self.__processor is not None:
            sub_rewards = self.__processor.get_rewards()
            re = [rew]
            if rew is not None:
                for r in sub_rewards:
                    re.append(r)
            return re
        else:
            return rew

    def _step(self, act):
        for i in range(len(self.__dis_act)):
            if act == self.__dis_act[i]:
                act = 0

        if not self.__loss_of_life_termination and not self.__loss_of_life_negative_reward:
            if not self.__is_terminal:
                next_lives = self.__ale.lives()
                if next_lives < self.__current_lives:
                    act = 1
                    self.__current_lives = next_lives
            return self._pre_step(act)
        else:
            rew = self._pre_step(act)
            next_lives = self.__ale.lives()
            if next_lives < self.__current_lives:
                if self.__loss_of_life_negative_reward:
                    rew -= 1
                self.__current_lives = next_lives
                self.__is_life_lost = True

            return rew

    def get_state(self):
        if not self.__max_2_frames:
            if self.__processor is not None:
                return self.__current_state
            else:
                return self.__current_buffer
        else:
            return self.__current_state

    def is_terminal(self):
        if self.__loss_of_life_termination and self.__is_life_lost:
            return True
        elif self.__max_episode_steps is not None and self.__current_steps > self.__max_episode_steps:
            return True
        else:
            return self.__is_terminal

    @staticmethod
    def get_rom_path(rom=None):
        if rom is None:
            return os.path.dirname(os.path.abspath(__file__)) + "/roms/"
        else:
            return os.path.dirname(os.path.abspath(__file__)) + "/roms/" + rom + ".bin"

    @staticmethod
    def list_all_roms():
        return [f for f in listdir(ALEEnvironment.get_rom_path()) if isfile(join(ALEEnvironment.get_rom_path(), f))]

    def get_state_space(self):
        if self.__processor is None:
            shape = self.__current_buffer.shape
        else:
            shape = self.__processor.process(self.__current_buffer).shape
        min_value = np.zeros(shape, dtype=np.uint8)
        max_value = np.full(shape, 255)
        return Space(min_value, max_value, True)

    def get_action_space(self):
        if self.__action_reduction >= 1:
            return Space(0, self.__action_reduction - 1, True)
        else:
            return Space(0, len(self.__action_set)-1, True)

    def step(self, act):
        if isinstance(act, (list, np.ndarray)):
            if len(act) <= 0:
                raise ValueError('Empty action list !')
            act = act[0]
        self.__current_steps += 1
        act = self.__action_set[act]
        rew = self._step(act)
        return rew

    def get_current_steps(self):
        return self.__current_steps

    def is_atari(self):
        return True

    def is_render(self):
        return self.__is_render

    def get_number_of_objectives(self):
        if self.__processor is None:
            return 1
        else:
            return self.__processor.get_number_of_objectives()

    def get_number_of_agents(self):
        if self.__processor is None:
            return 1
        else:
            return self.__processor.get_number_of_agents()

    def get_state_processor(self):
        return self.__processor


if __name__ == '__main__':
    def save_state(state, index):
        root_path = './states/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        full_path = join(root_path, 'state_{}.png'.format(index))
        img = Image.fromarray(state, 'L')
        img.save(full_path)

    env = ALEEnvironment(rom_name=ALEEnvironment.BREAKOUT,
                         is_render=True,
                         bitwise_max_on_two_consecutive_frames=True,
                         frame_skip=20,
                         loss_of_life_termination=True)

    max_action = env.get_action_space().get_max()
    min_action = env.get_action_space().get_min()
    print('Min action = {}, max action = {}'.format(min_action, max_action))

    env.reset()
    num_of_steps = 1000
    for i in range(num_of_steps):
        action = np.random.randint(0, max_action + 1)
        reward = env.step(action)
        if reward > 0:
            print("Reward:" + str(reward))
        next_state = env.get_state()
        save_state(next_state, i)
        terminal = env.is_terminal()
        if terminal:
            env.reset()

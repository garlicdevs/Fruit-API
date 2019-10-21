import numpy as np
import pygame
import os
import sys
import collections as cl
from fruit.envs.games.utils import Utils
from fruit.envs.games.mountain_car.sprites import GoalSprite, CarSprite
from fruit.envs.games.mountain_car.constants import MountainCarConstants


class MountainCar(object):
    def __init__(self, min_pos=-1.2, max_pos=0.6, min_vel=-0.07, max_vel=0.07, goal_pos=0.5, acceleration=0.001,
                 gravity_factor=-0.0025, hill_peak_freq=3.0, default_init_pos=-0.5, default_init_vel=0.0,
                 reward_per_step=-1, reward_at_goal=0, random_starts=False, transition_noise=0., seed=None,
                 render=True, speed=60, is_debug=False, frame_skip=1, friction=0, graphical_state=False,
                 discrete_states=6):
        self.vel = default_init_vel
        self.pos = default_init_pos
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.goal_pos = goal_pos
        self.acc = acceleration
        self.gra = gravity_factor
        self.hill_peak_freq = hill_peak_freq
        self.default_init_pos = default_init_pos
        self.default_init_vel = default_init_vel
        self.reward_per_step = reward_per_step
        self.reward_at_goal = reward_at_goal
        self.rand_starts = random_starts
        self.trans_noise = transition_noise
        self.last_action = 0
        self.rd = render
        self.speed = speed

        self.num_of_actions = 3
        self.num_of_objs = 3
        self.screen_size = 500
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.car_width = 60
        self.car_height = 30
        self.sprites = pygame.sprite.Group()
        self.frames_count = 0
        self.max_frames = 10000
        self.is_debug = is_debug
        self.frame_speed = 0
        self.log_freq = 500
        self.started_time = Utils.get_current_time()
        self.frame_skip = frame_skip
        self.current_buffer = np.array([[[0, 0, 0] for _ in range(self.screen_size)] for _ in range(self.screen_size)])
        self.mountain = []
        self.surface = pygame.Surface((self.screen_size, self.screen_size))
        self.friction = friction
        self.rewards_2 = cl.deque(maxlen=100)
        self.rewards_3 = cl.deque(maxlen=100)
        self.total_score = 0
        self.total_score_2 = 0
        self.total_score_3 = 0
        self.graphical_state = graphical_state
        self.max_states_per_dim = discrete_states

        if seed is None or seed < 0 or seed >= 9999:
            if seed is not None and (seed < 0 or seed >= 9999):
                print("Invalid seed ! Default seed = randint(0, 9999")
            self.seed = np.random.randint(0, 9999)
            self.random_seed = True
        else:
            self.random_seed = False
            self.seed = seed
            np.random.seed(seed)

        # Initialize
        self.__init_pygame_engine()

        # Create mountain
        self.__draw_sine_wave()

        # Create player
        self.__generate_car()

        # Create goal
        self.__generate_goal()

        # Render the first frame
        self.__render()

    def __draw_score(self):

        total_score = self.font.render(str(self.total_score) + " " + str(self.total_score_2) + " " +
                                       str(self.total_score_3),
                                       False, Utils.get_color(Utils.BLACK))
        self.screen.blit(total_score, (self.screen_size/2 - total_score.get_width()/2,
                                       self.screen_size - total_score.get_height()*2))

    @staticmethod
    def get_game_name():
        return "MOUNTAIN CAR"

    def get_pareto_solutions(self):
        return None

    def __check_reward(self):
        if self.__is_in_goal_region():
            r1 = self.reward_at_goal
        else:
            r1 = self.reward_per_step

        if len(self.rewards_2) > 0:
            r2 = self.rewards_2.pop()
        else:
            r2 = 0

        if len(self.rewards_3) > 0:
            r3 = self.rewards_3.pop()
        else:
            r3 = 0

        return [r1, r2, r3]

    def __is_in_goal_region(self):
        if self.pos >= self.goal_pos:
            return True
        else:
            return False

    def __get_height_at(self, pos):
        return -np.sin(self.hill_peak_freq * pos)

    def __get_slope(self, pos):
        return np.cos(self.hill_peak_freq * pos)

    def clone(self):
        if self.random_seed:
            seed = np.random.randint(0, 9999)
        else:
            seed = self.seed
        return MountainCar(min_pos=self.min_pos, max_pos=self.max_pos, min_vel=self.min_vel,
                           max_vel=self.max_vel, goal_pos=self.goal_pos, acceleration=self.acc,
                           gravity_factor=self.gra, hill_peak_freq=self.hill_peak_freq,
                           default_init_pos=self.default_init_pos, default_init_vel=self.default_init_vel,
                           reward_per_step=self.reward_per_step, reward_at_goal=self.reward_at_goal,
                           random_starts=self.rand_starts, transition_noise=self.trans_noise, seed=seed,
                           render=self.rd, speed=self.speed, is_debug=self.is_debug, frame_skip=self.frame_skip,
                           friction=self.friction, graphical_state=self.graphical_state,
                           discrete_states=self.max_states_per_dim)

    def get_num_of_objectives(self):
        return self.num_of_objs

    def get_seed(self):
        return self.seed

    def __init_pygame_engine(self):
        # Center the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Init Pygame engine
        pygame.init()

        self.font = pygame.font.Font(self.current_path + "/../../common/fonts/font.ttf", 15)

        if self.rd:
            pygame.display.set_caption(MountainCar.get_game_name())
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        else:
            self.screen = pygame.Surface((self.screen_size, self.screen_size))

    def __generate_car(self):
        image = pygame.image.load(self.current_path + "/graphics/car.png")
        if self.rd:
            image = pygame.transform.scale(image, (self.car_width, self.car_height)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.car_width, self.car_height))

        pos_x = self.__to_pixel_x(self.pos) - self.car_width/2
        pos_y = self.__to_pixel_y(self.__get_height_at(self.pos)) - self.car_height
        self.car = CarSprite(pos_x=pos_x, pos_y=pos_y, sprite_bg=image)
        self.sprites.add(self.car)

    def __generate_goal(self):
        flag_width = 25
        flag_height = 40
        image = pygame.image.load(self.current_path + "/graphics/flag.png")
        if self.rd:
            image = pygame.transform.scale(image, (flag_width, flag_height)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (flag_width, flag_height))

        pos_x = self.__to_pixel_x(self.goal_pos) - flag_width/2
        pos_y = self.__to_pixel_y(self.__get_height_at(self.goal_pos)) - flag_height + 15
        self.goal = GoalSprite(pos_x=pos_x, pos_y=pos_y, sprite_bg=image)
        self.sprites.add(self.goal)

    @staticmethod
    def __is_key_pressed():
        keys = pygame.key.get_pressed()
        for i in range(len(keys)):
            if keys[i] != 0:
                return i
        return -1

    def __to_real_x(self, pos):
        dist = self.max_pos - self.min_pos
        dist_pixels = self.screen_size
        actual_pos = pos * dist / dist_pixels + self.min_pos
        return actual_pos

    def __to_real_y(self, pos):
        dist = 2
        dist_pixels = self.screen_size-240
        actual_pos = (pos-60) * dist / dist_pixels + self.min_pos
        return actual_pos

    def __to_pixel_x(self, pos):
        dist = self.max_pos - self.min_pos
        dist_pixels = self.screen_size
        actual_pos = (pos - self.min_pos) * dist_pixels / dist
        return actual_pos

    def __to_pixel_y(self, pos):
        dist = 2
        dist_pixels = self.screen_size-240
        actual_pos = (pos - self.min_pos) * dist_pixels / dist + 60
        return actual_pos

    def __draw_sine_wave(self):
        self.surface.fill((255, 255, 255))
        pos = self.__to_real_x(0)
        prev_x = 0
        prev_y = self.__to_pixel_y(self.__get_height_at(pos))

        for x in range(1, self.screen_size):
            real_x = self.__to_real_x(x)
            real_y = self.__get_height_at(real_x)
            y = self.__to_pixel_y(real_y) + 15
            pygame.draw.line(self.surface, (200, 200, 100), (prev_x, prev_y), (x, y), 2)
            prev_x = x
            prev_y = y

        self.screen.blit(self.surface, (0, 0))

    def __human_control(self, key):
        action = -1
        if key == pygame.K_LEFT:
            self.move(MountainCarConstants.LEFT_ACTION)
            action = MountainCarConstants.LEFT_ACTION
        if key == pygame.K_RIGHT:
            self.move(MountainCarConstants.RIGHT_ACTION)
            action = MountainCarConstants.RIGHT_ACTION
        if key == pygame.K_UP:
            self.move(MountainCarConstants.UP_ACTION)
            action = MountainCarConstants.UP_ACTION

        return action

    def move(self, action):

        self.last_action = action

        noise = 2.0 * self.acc * self.trans_noise * (np.random.rand() - 0.5)
        self.vel = self.vel + (noise + (action - 1) * self.acc) + self.__get_slope(self.pos) * self.gra
        if self.vel > 0:
            self.vel = self.vel - self.friction
        elif self.vel < -0:
            self.vel = self.vel + self.friction
        else:
            self.vel = 0
        if self.vel > self.max_vel:
            self.vel = self.max_vel
        if self.vel < self.min_vel:
            self.vel = self.min_vel
        self.pos = self.pos + self.vel
        if self.pos > self.max_pos:
            self.pos = self.max_pos
        if self.pos < self.min_pos:
            self.pos = self.min_pos
        if self.pos == self.min_pos and self.vel < 0:
            self.vel = 0

        pos_x = self.__to_pixel_x(self.pos) - self.car_width / 2
        pos_y = self.__to_pixel_y(self.__get_height_at(self.pos)) - self.car_height*0.9

        p1 = self.pos
        h1 = self.__get_height_at(p1)
        p2 = self.pos + 0.05
        h2 = self.__get_height_at(p2)

        tan_alpha = (h2 - h1)/0.05
        alpha = np.arctan(tan_alpha)
        alpha = -alpha * 180/3.1415

        if -50 < alpha < 50:
            alpha = 0
        elif alpha >= 50:
            alpha = 50
        elif alpha < -50:
            alpha = -50

        dist = self.car.alpha - alpha
        if dist < 0:
            dist = -dist

        if dist >= 50:
            if alpha == 0:
                self.car.restore()
            else:
                self.car.rotate(alpha)

        self.car.rect.x = pos_x
        self.car.rect.y = pos_y

        if alpha == 0:
            self.car.rect.y = pos_y + 20

    def __handle_event(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.reset()
                sys.exit()

        key = MountainCar.__is_key_pressed()
        if key >= 0:
            return self.__human_control(key)

        return -1

    def get_key_pressed(self):
        return MountainCar.__is_key_pressed()

    def __calculate_fps(self):
        self.frames_count = self.frames_count + 1
        if self.max_frames > 0:
            if self.frames_count > self.max_frames:
                self.end_of_game = True
        current_time = Utils.get_current_time()
        if current_time > self.started_time:
            self.frame_speed = self.frames_count / (current_time - self.started_time)
        else:
            self.frame_speed = 0

    def __print_info(self):
        if self.is_debug:
            if self.frames_count % self.log_freq == 0:
                print("Current frame:", self.frames_count)
                print("Total score:", self.total_score)
                print("Frame speed (FPS):", self.frame_speed)
                print("")

    def __render(self, is_agent=False):
        human_action = -1
        if self.rd:
            human_action = self.__handle_event()
        if not is_agent:
            if human_action != -1:
                if human_action == MountainCarConstants.LEFT_ACTION:
                    self.rewards_2.append(-1)
                    self.total_score_2 = self.total_score_2 - 1
                elif human_action == MountainCarConstants.RIGHT_ACTION:
                    self.rewards_3.append(-1)
                    self.total_score_3 = self.total_score_3 - 1
                self.total_score = self.total_score + self.reward_per_step
                self.move(human_action)
            else:
                self.total_score = self.total_score + self.reward_per_step
                self.move(MountainCarConstants.UP_ACTION)

        self.__draw_sine_wave()

        self.__draw_score()

        # Update sprites
        self.sprites.update()

        # Redraw all sprites
        self.sprites.draw(self.screen)

        # Show to the screen what we're have drawn so far
        if self.rd:
            pygame.display.flip()

        # Calculate fps
        self.__calculate_fps()

        # Debug
        self.__print_info()

        # Maintain 20 fps
        pygame.time.Clock().tick(self.speed)

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):

        for sprite in self.sprites:
            sprite.kill()

        self.pos = self.default_init_pos
        self.vel = self.default_init_vel
        if self.rand_starts:
            rand_start_pos = self.default_init_pos + 0.25 * (np.random.rand() - 0.5)
            self.pos = rand_start_pos
            rand_start_vel = self.default_init_vel + 0.25 * (np.random.rand() - 0.5)
            self.vel = rand_start_vel

        self.__generate_car()

        self.__generate_goal()

        if self.is_debug:
            interval = Utils.get_current_time() - self.started_time
            print("#################  RESET GAME  ##################")
            print("Episode terminated after:", interval, "(s)")
            print("Total score:", self.total_score)
            print("#################################################")

        self.rewards_2.clear()
        self.rewards_3.clear()
        self.total_score = 0
        self.total_score_2 = 0
        self.total_score_3 = 0
        self.__render()

    def step(self, action):

        if action == MountainCarConstants.LEFT_ACTION:
            self.rewards_2.append(-1)
            self.total_score_2 = self.total_score_2 - 1
        elif action == MountainCarConstants.RIGHT_ACTION:
            self.rewards_3.append(-1)
            self.total_score_3 = self.total_score_3 - 1
        self.total_score = self.total_score + self.reward_per_step

        self.move(action)

        if self.frame_skip <= 1:
            self.__render(True)
        else:
            self.__render(True)
            for _ in range(self.frame_skip-1):
                self.move(action)
                self.__render(True)

        return self.__check_reward()

    def render(self, is_agent=False):
        self.__render(is_agent)

    def step_all(self, action):
        r = self.step(action)
        next_state = self.get_state()
        terminal = self.is_terminal()
        return next_state, r, terminal

    def get_state_space(self):
        if self.graphical_state:
            return [self.screen_size, self.screen_size]
        else:
            from fruit.types.priv import Space
            return Space(0, self.max_states_per_dim * self.max_states_per_dim - 1, True)

    def get_action_space(self):
        return range(self.num_of_actions)

    def get_state(self):

        if self.graphical_state:
            pygame.pixelcopy.surface_to_array(self.current_buffer, self.screen)
            return self.current_buffer
        else:
            pos_index = np.floor((self.pos - self.min_pos) * self.max_states_per_dim / (self.max_pos - self.min_pos))
            vel_index = np.floor((self.vel - self.min_vel) * self.max_states_per_dim / (self.max_vel - self.min_vel))
            return int(pos_index * self.max_states_per_dim + vel_index)

    def is_terminal(self):
        if self.pos > self.goal_pos:
            return True
        else:
            return False

    def debug(self):
        self.__print_info()

    def get_num_of_actions(self):
        return self.num_of_actions

    def is_render(self):
        return self.rd


def check_map():
    from PIL import Image
    from pathlib import Path
    home = str(Path.home())
    game = MountainCar(render=False, graphical_state=True)
    num_of_actions = game.get_num_of_actions()
    game.reset()
    full_path = home + '/Desktop/Images/'
    count = 0

    while True:
        random_action = np.random.randint(0, num_of_actions)
        reward = game.step(random_action)
        print(reward)
        next_state = Utils.process_state(game.get_state())
        is_terminal = game.is_terminal()

        # save state and map
        count = count + 1
        img = Image.fromarray(next_state, 'L')
        img.save(full_path + str(count) +'.png')

        if is_terminal:
            print("Total Score", game.total_score)
            game.reset()
            break


if __name__ == '__main__':

    game = MountainCar(render=True, speed=30)

    while True:
        game.render()

        if game.is_terminal():
            break

    # check_map()
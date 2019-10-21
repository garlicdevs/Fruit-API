from fruit.envs.games.milk_factory.constants import GlobalConstants
from fruit.envs.games.milk_factory.manager import ResourceManager
from fruit.envs.games.milk_factory.sprites import MilkRobotSprite, FixRobotSprite, MilkSprite, StatusMilkSprite, \
    StatusErrorSprite
from fruit.envs.games.milk_factory.stages import StageMap
from fruit.envs.games.utils import Utils
import numpy as np
import pygame
import os
import sys
import collections as cl


class MilkFactory(object):

    def __init__(self, render=False, speed=600, max_frames=1000, frame_skip=1, number_of_milk_robots=3,
                 number_of_fix_robots=2, number_of_milks=1, seed=None, human_control=True, error_freq=0.01,
                 human_control_robot=0, milk_speed=3, debug=False, action_combined_mode=False, show_status=False,
                 number_of_exits=1):

        # Prepare internal data
        self.num_of_objs = number_of_fix_robots + number_of_milk_robots

        self.screen_size = GlobalConstants.SCREEN_SIZE
        self.tile_size = GlobalConstants.TILE_SIZE
        self.max_frames = max_frames
        self.rd = render
        self.screen = None
        self.speed = speed

        self.sprites = pygame.sprite.Group()
        self.conveyor_sprites = pygame.sprite.Group()
        self.milk_sprites = pygame.sprite.Group()
        self.robot_sprites = pygame.sprite.Group()
        self.target_sprites = pygame.sprite.Group()

        self.robots = []

        self.number_of_milk_robots = number_of_milk_robots
        self.number_of_fix_robots = number_of_fix_robots
        self.number_of_milks = number_of_milks

        self.num_of_actions = GlobalConstants.NUM_OF_ACTIONS
        self.num_of_tiles = int(self.screen_size/self.tile_size)

        self.end_of_game = False
        self.is_debug = debug

        self.frames_count = 0
        self.total_score = [0 for _ in range(self.num_of_objs)]

        self.milk_speed = milk_speed
        self.show_status = show_status

        self.font_size = GlobalConstants.FONT_SIZE
        self.human_control = human_control
        self.human_control_robot = human_control_robot
        self.num_of_exits = number_of_exits

        self.log_freq = 60
        if self.log_freq == 0:
            self.log_freq = 60

        self.error_freq = error_freq

        self.current_path = os.path.dirname(os.path.abspath(__file__))

        self.current_stage = number_of_exits-1

        self.current_buffer = np.array([[[0, 0, 0] for _ in range(self.screen_size + int(self.tile_size))]
                                        for _ in range(self.screen_size)])

        self.map_buffer = np.array([[[0, 0, 0] for _ in range(self.screen_size)]
                                    for _ in range(self.screen_size)])

        self.map = None
        self.map_sprites = [[] for _ in range(self.number_of_milk_robots)]
        self.map_sprites_status = [[] for _ in range(self.number_of_milk_robots)]

        self.pareto_solutions = None
        self.frame_speed = 0
        self.frame_skip = frame_skip
        self.started_time = Utils.get_current_time()
        self.steps = 0

        self.rewards = [cl.deque(maxlen=100) for _ in range(self.num_of_objs)]

        self.action_combined_mode = action_combined_mode

        if self.human_control:
            if not self.rd:
                raise ValueError("Invalid parameter ! Human control must be in rendering mode")

        # Seed is used to generate a stochastic environment
        if seed is None or seed < 0 or seed >= 9999:
            self.seed = np.random.randint(0, 9999)
            self.random_seed = True
        else:
            self.random_seed = False
            self.seed = seed
            np.random.seed(seed)

        # Initialize
        self.__init_pygame_engine()

        # Load map
        self.stage_map.load_map(self.current_stage, self.conveyor_sprites, self.target_sprites)

        # Create milks
        self.__generate_milks()

        # Create robots
        self.__generate_robots()

        # Render communication map
        self.__init_map()

        # Render the first frame
        self.__render()

    def __init_map(self):
        max_index = int(self.screen_size/self.tile_size)

        for i in range(len(self.map_sprites)):
            busy_status = StatusMilkSprite(self.tile_size, i, 0, self.rc_manager.get_image(ResourceManager.MILK_TILE))
            busy_group = pygame.sprite.Group()
            busy_group.add(busy_status)
            self.map_sprites[i].append(busy_group)

            error_status = StatusErrorSprite(self.tile_size, i, 0, self.rc_manager.get_image(ResourceManager.ERROR_TILE))
            error_group = pygame.sprite.Group()
            error_group.add(error_status)
            self.map_sprites[i].append(error_group)

        for i in range(len(self.map_sprites_status)):
            busy_status = StatusMilkSprite(self.tile_size, max_index-1 - i, max_index, self.rc_manager.get_image(ResourceManager.MILK_TILE))
            busy_group = pygame.sprite.Group()
            busy_group.add(busy_status)
            self.map_sprites_status[i].append(busy_group)

            error_status = StatusErrorSprite(self.tile_size, max_index-1-i, max_index, self.rc_manager.get_image(ResourceManager.ERROR_TILE))
            error_group = pygame.sprite.Group()
            error_group.add(error_status)
            self.map_sprites_status[i].append(error_group)

    def get_num_of_agents(self):
        return self.number_of_milk_robots + self.number_of_fix_robots

    @staticmethod
    def get_game_name():
        return "MILK FACTORY"

    def clone(self):
        if self.random_seed:
            seed = np.random.randint(0, 9999)
        else:
            seed = self.seed
        return MilkFactory(render=self.rd, speed=self.speed, max_frames=self.max_frames, frame_skip=self.frame_skip,
                           number_of_milk_robots=self.number_of_milk_robots,
                           number_of_fix_robots=self.number_of_fix_robots,
                           number_of_milks=self.number_of_milks, seed=seed, human_control=self.human_control,
                           error_freq=self.error_freq,
                           human_control_robot=self.human_control_robot,
                           milk_speed=self.milk_speed,
                           debug=self.is_debug,
                           action_combined_mode=self.action_combined_mode,
                           show_status=self.show_status,
                           number_of_exits=self.num_of_exits)

    def get_num_of_objectives(self):
        return self.num_of_objs

    def get_seed(self):
        return self.seed

    def __init_pygame_engine(self):
        # Center the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Init Pygame engine
        pygame.init()

        # Init joysticks
        self.num_of_joysticks = pygame.joystick.get_count()
        self.joystick_p1 = None
        if self.num_of_joysticks > 0:
            self.joystick_p1 = pygame.joystick.Joystick(0)
            self.joystick_p1.init()

        if self.rd:
            pygame.display.set_caption(MilkFactory.get_game_name())
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + int(self.tile_size)))
        else:
            self.screen = pygame.Surface((self.screen_size, self.screen_size + int(self.tile_size)))

        self.map = pygame.Surface((self.screen_size, self.screen_size))

        self.rc_manager = ResourceManager(current_path=self.current_path, font_size=self.font_size,
                                          tile_size=self.tile_size, is_render=self.rd)
        self.font = self.rc_manager.get_font()
        self.stage_map = StageMap(self.num_of_tiles, tile_size=self.tile_size, current_path=self.current_path,
                                  sprites=self.sprites,
                                  resources_manager=self.rc_manager)

    def __generate_robots(self):
        for _ in range(self.number_of_milk_robots):
            pos_x = np.random.randint(0, self.num_of_tiles)
            pos_y = np.random.randint(1, self.num_of_tiles)

            robot = MilkRobotSprite(self.tile_size, pos_x=pos_x, pos_y=pos_y, error_freq=self.error_freq,
                                    sprite_bg=(self.rc_manager.get_image(ResourceManager.ROBOT_MILK),
                                               self.rc_manager.get_image(ResourceManager.ROBOT_MILK_ERROR),
                                               self.rc_manager.get_image(ResourceManager.ROBOT_MILK_PICKED)))

            self.robots.append(robot)
            self.robot_sprites.add(robot)
            self.sprites.add(robot)

        for _ in range(self.number_of_fix_robots):
            pos_x = np.random.randint(0, self.num_of_tiles)
            pos_y = np.random.randint(1, self.num_of_tiles)
            robot = FixRobotSprite(self.tile_size, pos_x=pos_x, pos_y=pos_y,
                                   sprite_bg=self.rc_manager.get_image(ResourceManager.ROBOT_FIX))

            self.robots.append(robot)
            self.robot_sprites.add(robot)
            self.sprites.add(robot)

    def __generate_milks(self):
        for i in range(self.number_of_milks):
            pos_x = i
            pos_y = 0

            milk = MilkSprite(self.tile_size, pos_x=pos_x, pos_y=pos_y, speed=self.milk_speed,
                              sprite_bg=self.rc_manager.get_image(ResourceManager.MILK_TILE))

            self.sprites.add(milk)
            self.milk_sprites.add(milk)

    def __draw_score(self):
        score_str = 'Score: '
        for i in range(self.num_of_objs):
            score_str += str(self.total_score[i])
            if i < self.num_of_objs - 1:
                score_str += ','

        total_score = self.font.render(score_str, False, Utils.get_color(Utils.WHITE))
        self.screen.blit(total_score, (10, self.screen_size + total_score.get_height()/1.3))

    @staticmethod
    def __is_key_pressed():
        keys = pygame.key.get_pressed()
        for i in range(len(keys)):
            if keys[i] != 0:
                return i
        return -1

    def __human_control(self, key):
        if self.human_control:
            robot = self.robots[self.human_control_robot]
            if key == pygame.K_LEFT:
                robot.move(GlobalConstants.LEFT_ACTION, self.sprites)
            if key == pygame.K_RIGHT:
                robot.move(GlobalConstants.RIGHT_ACTION, self.sprites)
            if key == pygame.K_UP:
                robot.move(GlobalConstants.UP_ACTION, self.sprites)
            if key == pygame.K_DOWN:
                robot.move(GlobalConstants.DOWN_ACTION, self.sprites)
            if key == pygame.K_a:
                reward = robot.move(GlobalConstants.FIRE_ACTION, self.sprites)
                if reward > 0:
                    self.rewards[self.human_control_robot].append(reward)
                    self.total_score[self.human_control_robot] += reward

    def __handle_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.reset()
                sys.exit()

        if not self.human_control:
            return True

        key = MilkFactory.__is_key_pressed()
        if key >= 0:
            self.__human_control(key)

        return True

    @staticmethod
    def get_key_pressed():
        return MilkFactory.__is_key_pressed()

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

    def __check_goal(self):
        return

    def __update_map(self):
        self.map.fill(Utils.get_color(Utils.BLACK))

        index = 0
        for robot in self.robots:
            if type(robot) is MilkRobotSprite:
                if robot.error:
                    self.map_sprites[index][1].draw(self.map)
                elif robot.picked:
                    self.map_sprites[index][0].draw(self.map)
                index += 1

    def __show_status(self):
        if not self.show_status:
            return
        index = 0
        for robot in self.robots:
            if type(robot) is MilkRobotSprite:
                if robot.error:
                    self.map_sprites_status[index][1].draw(self.screen)
                elif robot.picked:
                    self.map_sprites_status[index][0].draw(self.screen)
                index += 1

    def __render(self):

        # Handle user event
        if self.rd:
            self.__handle_event()

        # Draw background first
        self.screen.fill(Utils.get_color(Utils.BLACK))

        # Update sprites
        self.sprites.update()

        # Redraw all sprites
        self.sprites.draw(self.screen)

        # Redraw conveyors
        self.conveyor_sprites.draw(self.screen)

        # Redraw milks
        self.milk_sprites.draw(self.screen)

        # Redraw target
        self.target_sprites.draw(self.screen)

        # Redraw robots
        self.robot_sprites.draw(self.screen)

        # Update map
        self.__update_map()

        # Show status
        self.__show_status()

        # Draw score
        self.__draw_score()

        # Check terminal state
        self.__check_goal()

        # Show to the screen what we're have drawn so far
        if self.rd:
            pygame.display.flip()

        # Maintain 20 fps
        pygame.time.Clock().tick(self.speed)

        # Calculate fps
        self.__calculate_fps()

        # Debug
        self.__print_info()

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):
        global global_steps
        global_steps = 0

        self.end_of_game = False
        self.frames_count = 0
        self.started_time = Utils.get_current_time()

        for sprite in self.conveyor_sprites:
            sprite.kill()
        self.conveyor_sprites.empty()

        for sprite in self.milk_sprites:
            sprite.kill()
        self.milk_sprites.empty()

        for sprite in self.target_sprites:
            sprite.kill()
        self.target_sprites.empty()

        for sprite in self.robot_sprites:
            sprite.kill()
        self.robot_sprites.empty()

        for sprite in self.sprites:
            sprite.kill()
        self.sprites.empty()

        self.robots.clear()

        self.stage_map.load_map(self.current_stage, self.conveyor_sprites, self.target_sprites)

        # Create milks
        self.__generate_milks()

        # Create players
        self.__generate_robots()

        if self.is_debug:
            interval = Utils.get_current_time() - self.started_time
            print("#################  RESET GAME  ##################")
            print("Episode terminated after:", interval, "(s)")
            print("Total score:", self.total_score)
            print("#################################################")

        self.total_score = [0 for _ in range(self.num_of_objs)]

        self.__render()

    def __check_reward(self):
        r = [0 for _ in range(self.num_of_objs)]
        for i in range(self.num_of_objs):
            rewards = self.rewards[i]
            if len(rewards) <= 0:
                r[i] = 0
            else:
                r[i] = rewards.pop()

        return r

    def __fire(self):
        return

    def step(self, actions):
        if self.action_combined_mode:
            t = actions
            r = actions % GlobalConstants.NUM_OF_ACTIONS
            d = int(actions / GlobalConstants.NUM_OF_ACTIONS)
            actions = [0 for _ in range(self.num_of_objs)]
            actions[d] = r
        if self.human_control:
            raise ValueError("Error: human control mode")
        if not self.human_control:
            for i in range(self.num_of_objs):
                action = actions[i]
                robot = self.robots[i]

                if action == GlobalConstants.LEFT_ACTION:
                    robot.move(GlobalConstants.LEFT_ACTION, self.sprites)
                elif action == GlobalConstants.RIGHT_ACTION:
                    robot.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                elif action == GlobalConstants.UP_ACTION:
                    robot.move(GlobalConstants.UP_ACTION, self.sprites)
                elif action == GlobalConstants.DOWN_ACTION:
                    robot.move(GlobalConstants.DOWN_ACTION, self.sprites)
                elif action == GlobalConstants.FIRE_ACTION:
                    reward = robot.move(GlobalConstants.FIRE_ACTION, self.sprites)
                    if reward > 0:
                        self.rewards[i].append(reward)
                        self.total_score[i] += reward

        if self.frame_skip <= 1:
            self.__render()
        else:
            for _ in range(self.frame_skip):
                self.__render()

        return self.__check_reward()

    def render(self):
        self.__render()

    def step_all(self, action):
        r = self.step(action)
        next_state = self.get_state()
        terminal = self.is_terminal()
        return next_state, r, terminal

    def get_state_space(self):
        return [self.screen_size, self.screen_size]

    def get_action_space(self):
        if self.action_combined_mode:
            return range(self.num_of_actions * self.num_of_objs)
        else:
            return range(self.num_of_actions)

    def get_state(self):
        pygame.pixelcopy.surface_to_array(self.current_buffer, self.screen)
        return self.current_buffer

    def get_map(self, type=-1):
        pygame.pixelcopy.surface_to_array(self.map_buffer, self.map)
        return self.map_buffer

    def is_terminal(self):
        return self.end_of_game

    def debug(self):
        self.__print_info()

    def get_num_of_actions(self):
        if self.action_combined_mode:
            return self.num_of_objs * self.num_of_actions
        else:
            return self.num_of_actions

    def is_render(self):
        return self.rd


def check_control(human_control=True):
    from PIL import Image
    from pathlib import Path
    from os.path import isdir
    home = str(Path.home())
    full_path = home + '/Desktop/Images/'
    print(full_path)
    count = 0

    game = MilkFactory(render=True, speed=10, human_control=human_control, number_of_milk_robots=2,
                       number_of_fix_robots=1, error_freq=0.01, number_of_milks=2, milk_speed=3, number_of_exits=2,
                       human_control_robot=0, debug=True, action_combined_mode=False, show_status=True)
    game.reset()
    num_of_agents = game.get_num_of_agents()

    while True:
        if not human_control:
            num_of_actions = game.get_num_of_actions()
            actions = []
            for i in range(num_of_agents):
                random_action = np.random.randint(0, num_of_actions)
                actions.append(random_action)

            print(actions)
            reward = game.step(actions)
            print(reward)
        else:
            game.render()

        next_state = Utils.process_state(game.get_state())
        next_map = Utils.process_state(game.get_map())

        # save state and map
        count = count + 1
        if isdir(full_path):
            img = Image.fromarray(next_state, 'L')
            img.save(full_path + str(count) + '_state.png')
            img = Image.fromarray(next_map, 'L')
            img.save(full_path + str(count) + '_map.png')

        is_terminal = game.is_terminal()

        if is_terminal:
            print("Total Score", game.total_score)
            break


if __name__ == '__main__':
    check_control()
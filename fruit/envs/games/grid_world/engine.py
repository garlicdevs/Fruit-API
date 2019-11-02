from fruit.envs.games.grid_world.constants import GlobalConstants
from fruit.envs.games.grid_world.manager import ResourceManager
from fruit.envs.games.grid_world.sprites import CharacterSprite
from fruit.envs.games.grid_world.stages import StageMap
from fruit.envs.games.utils import Utils
import numpy as np
import pygame
import os
import sys
import collections as cl

# Thanks to Daniel Eddeland for providing the game graphics
# https://opengameart.org/content/lpc-farming-tilesets-magic-animations-and-ui-elements


class GridWorld(object):
    def __init__(self, render=False, speed=60, max_frames=100000, frame_skip=1, graphical_state=True,
                 seed=None, num_of_obstacles=2, number_of_rows=4, number_of_columns=4, debug=False, stage=0,
                 agent_start_x=0, agent_start_y=0):

        self.num_of_rows = number_of_rows
        self.num_of_columns = number_of_columns
        self.screen_width = GlobalConstants.TILE_SIZE * number_of_columns
        self.screen_height = GlobalConstants.TILE_SIZE * number_of_rows
        self.tile_size = GlobalConstants.TILE_SIZE
        self.max_frames = max_frames
        self.rd = render
        self.screen = None
        self.speed = speed
        self.num_of_obstacles = num_of_obstacles

        self.sprites = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.minuses = pygame.sprite.Group()
        self.players = pygame.sprite.Group()
        self.keys = pygame.sprite.Group()
        self.lands = pygame.sprite.Group()

        self.graphical_state = graphical_state

        self.num_of_actions = GlobalConstants.NUM_OF_ACTIONS
        self.end_of_game = False
        self.is_debug = debug
        self.frames_count = 0
        self.total_score = 0
        self.font_size = GlobalConstants.FONT_SIZE
        self.log_freq = 60
        if self.log_freq == 0:
            self.log_freq = 60
        self.current_stage = stage
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.player_speed = GlobalConstants.PLAYER_SPEED
        self.current_buffer = np.array([[[0, 0, 0] for _ in range(self.screen_height + int(self.tile_size/2))]
                                        for _ in range(self.screen_width)])
        self.frame_speed = 0
        self.frame_skip = frame_skip
        self.started_time = Utils.get_current_time()
        self.next_rewards = cl.deque(maxlen=100)
        self.num_of_objs = 1
        self.steps = 0
        self.rewards = cl.deque(maxlen=100)
        self.agent_start_x = agent_start_x
        self.agent_start_y = agent_start_y

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
        self.stage_map.load_map(self.current_stage)

        # Create players
        self.__generate_players()

        # Render the first frame
        self.__render()

    @staticmethod
    def get_game_name():
        return "GRID WORLD"

    def clone(self):
        if self.random_seed:
            seed = np.random.randint(0, 9999)
        else:
            seed = self.seed

        return GridWorld(render=self.rd, speed=self.speed, max_frames=self.max_frames, frame_skip=self.frame_skip,
                         graphical_state=self.graphical_state, seed=seed, num_of_obstacles=self.num_of_obstacles,
                         number_of_rows=self.num_of_rows, number_of_columns=self.num_of_columns, debug=self.is_debug,
                         stage=self.current_stage, agent_start_x=self.agent_start_x, agent_start_y=self.agent_start_y)

    def get_num_of_objectives(self):
        return self.num_of_objs

    def get_seed(self):
        return self.seed

    def __init_pygame_engine(self):
        # Center the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Init Pygame engine
        pygame.init()

        if self.rd:
            pygame.display.set_caption(GridWorld.get_game_name())
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height + int(self.tile_size/2)))
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height + int(self.tile_size/2)))
        self.rc_manager = ResourceManager(current_path=self.current_path, font_size=self.font_size,
                                          tile_size=self.tile_size, is_render=self.rd)
        self.font = self.rc_manager.get_font()
        self.stage_map = StageMap(num_of_rows=self.num_of_rows, num_of_columns=self.num_of_columns,
                                  tile_size=self.tile_size, current_path=self.current_path, seed=self.seed,
                                  num_of_obstacles=self.num_of_obstacles, sprites=self.sprites, keys=self.keys,
                                  obstacles=self.obstacles, lands=self.lands, resources_manager=self.rc_manager,
                                  minuses=self.minuses)

    def __generate_players(self):
        pos_x = self.agent_start_x
        pos_y = self.agent_start_y

        self.player1 = CharacterSprite(self.tile_size, pos_x=pos_x, pos_y=pos_y,
                                       sprite_bg=((self.rc_manager.get_image(ResourceManager.GUY_L_1_TILE),
                                                   self.rc_manager.get_image(ResourceManager.GUY_L_2_TILE)),
                                                  (self.rc_manager.get_image(ResourceManager.GUY_R_1_TILE),
                                                   self.rc_manager.get_image(ResourceManager.GUY_R_2_TILE)),
                                                  (self.rc_manager.get_image(ResourceManager.GUY_U_1_TILE),
                                                   self.rc_manager.get_image(ResourceManager.GUY_U_2_TILE)),
                                                  (self.rc_manager.get_image(ResourceManager.GUY_D_1_TILE),
                                                   self.rc_manager.get_image(ResourceManager.GUY_D_2_TILE))),
                                       num_of_rows=self.num_of_rows, num_of_columns=self.num_of_columns
                                       )
        self.players.add(self.player1)

    def __draw_score(self):
        total_score = self.font.render('Score:' + str(self.total_score),
                                       False, Utils.get_color(Utils.WHITE))
        self.screen.blit(total_score, (10, self.screen_height + total_score.get_height()/1.3))

    def __handle_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.reset()
                sys.exit()

        return True

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
        for key in self.keys:
            if self.player1.pos_x == key.pos_x and self.player1.pos_y == key.pos_y:
                self.total_score = self.total_score + GlobalConstants.KEY_REWARD
                self.rewards.append(GlobalConstants.KEY_REWARD)
                self.end_of_game = True
                return
        for minus in self.minuses:
            if self.player1.pos_x == minus.pos_x and self.player1.pos_y == minus.pos_y:
                self.total_score = self.total_score + GlobalConstants.MINUS_REWARD
                self.rewards.append(GlobalConstants.MINUS_REWARD)

    def __render(self):
        # Handle user event
        if self.rd:
            self.__handle_event()

        # Draw background first
        self.screen.fill(Utils.get_color(Utils.BLACK))

        # Draw land
        self.lands.draw(self.screen)

        # Update sprites
        self.sprites.update()

        # Redraw all sprites
        self.sprites.draw(self.screen)

        # Update player
        self.players.update()

        # Draw player
        self.players.draw(self.screen)

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

        for sprite in self.sprites:
            sprite.kill()

        for player in self.players:
            player.kill()

        for apple in self.lands:
            apple.kill()

        for door in self.keys:
            door.kill()

        for tree in self.obstacles:
            tree.kill()

        for minus in self.minuses:
            minus.kill()

        for land in self.lands:
            land.kill()

        self.stage_map.load_map(self.current_stage)

        # Create players
        self.__generate_players()

        if self.is_debug:
            interval = Utils.get_current_time() - self.started_time
            print("#################  RESET GAME  ##################")
            print("Episode terminated after:", interval, "(s)")
            print("Total score:", self.total_score)
            print("#################################################")

        self.total_score = 0

        self.__render()

    def __check_reward(self):
        if len(self.rewards) <= 0:
            r1 = GlobalConstants.MOVE_REWARD
            self.total_score += r1
        else:
            r1 = self.rewards.pop()

        return r1

    def step(self, action):
        if action == GlobalConstants.LEFT_ACTION:
            self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
        elif action == GlobalConstants.RIGHT_ACTION:
            self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
        elif action == GlobalConstants.UP_ACTION:
            self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
        elif action == GlobalConstants.DOWN_ACTION:
            self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
        else:
            raise ValueError('Wrong action: {}!'.format(action))

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
        if self.graphical_state:
            return [self.screen_width, self.screen_height]
        else:
            from fruit.types.priv import Space
            return Space(0, self.num_of_rows * self.num_of_columns - 1, True)

    def get_action_space(self):
        return range(self.num_of_actions)

    def get_state(self):
        if self.graphical_state:
            pygame.pixelcopy.surface_to_array(self.current_buffer, self.screen)
            return self.current_buffer
        else:
            return self.player1.pos_y * self.num_of_columns + self.player1.pos_x

    def is_terminal(self):
        return self.end_of_game

    def debug(self):
        self.__print_info()

    def get_num_of_actions(self):
        return self.num_of_actions

    def is_render(self):
        return self.rd

    @staticmethod
    def get_num_of_agents():
        return 1


if __name__ == '__main__':
    game = GridWorld(render=True, frame_skip=1, num_of_obstacles=2, graphical_state=False, stage=1,
                     number_of_rows=8, number_of_columns=9, speed=1, seed=100, agent_start_x=2, agent_start_y=2)
    num_of_actions = game.get_num_of_actions()
    game.reset()
    state = game.get_state()

    for i in range(10000):
        random_action = np.random.randint(0, num_of_actions)
        reward = game.step(random_action)
        # next_state = Utils.process_state(game.get_state())
        next_state = game.get_state()
        is_terminal = game.is_terminal()
        state = next_state

        print('Action', random_action, 'Score Achieved', reward, 'Total Score', game.total_score, 'State', state)

        if is_terminal:
            print("Total Score", game.total_score)
            game.reset()
            break

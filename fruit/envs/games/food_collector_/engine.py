from fruit.envs.games.food_collector_.constants import GlobalConstants
from fruit.envs.games.food_collector_.manager import ResourceManager
from fruit.envs.games.food_collector_.stages import StageMap
from fruit.envs.games.food_collector_.sprites import DoorSprite, FoodSprite, CharacterSprite, AppleSprite
from fruit.envs.games.utils import Utils
import numpy as np
import pygame
import os
import sys
import collections as cl

# Thanks to Daniel Eddeland for providing the game graphics
# https://opengameart.org/content/lpc-farming-tilesets-magic-animations-and-ui-elements


class FoodCollector(object):

    def __init__(self, render=False, speed=60, max_frames=100000, frame_skip=5,
                 seed=None, num_of_apples=1, human_control=True, debug=False):

        # Prepare internal data
        self.screen_size = GlobalConstants.SCREEN_SIZE
        self.tile_size = GlobalConstants.TILE_SIZE
        self.max_frames = max_frames
        self.rd = render
        self.screen = None
        self.speed = speed
        self.num_of_apples = num_of_apples
        self.sprites = pygame.sprite.Group()
        self.apples = pygame.sprite.Group()
        self.players = pygame.sprite.Group()
        self.doors = pygame.sprite.Group()
        self.trees = pygame.sprite.Group()
        self.food = pygame.sprite.Group()
        self.lands = pygame.sprite.Group()
        self.num_of_actions = GlobalConstants.NUM_OF_ACTIONS
        self.num_of_tiles = int(self.screen_size/self.tile_size)
        self.end_of_game = False
        self.is_debug = debug
        self.frames_count = 0
        self.total_score = 0
        self.total_score2 = 0
        self.font_size = GlobalConstants.FONT_SIZE
        self.human_control = human_control
        self.log_freq = 60
        if self.log_freq == 0:
            self.log_freq = 60
        self.current_stage = 0
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.player_speed = GlobalConstants.PLAYER_SPEED
        self.current_buffer = np.array([[[0, 0, 0] for _ in range(self.screen_size + int(self.tile_size/2))] for _ in range(self.screen_size)])
        self.pareto_solutions = None
        self.frame_speed = 0
        self.frame_skip = frame_skip
        self.started_time = Utils.get_current_time()
        self.next_rewards = cl.deque(maxlen=100)
        self.num_of_objs = 2
        self.steps = 0
        self.life = 100
        self.life_fq = 10
        self.rewards = cl.deque(maxlen=100)
        self.rewards_eat = cl.deque(maxlen=100)

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
        self.stage_map.load_map(self.current_stage)

        # Create apples and food
        self.__generate_food()

        # Create door
        self.__generate_particles()

        # Create players
        self.__generate_players()

        # Render the first frame
        self.__render()

    @staticmethod
    def get_game_name():
        return "FOOD COLLECTOR"

    def clone(self):
        if self.random_seed:
            seed = np.random.randint(0, 9999)
        else:
            seed = self.seed
        return FoodCollector(render=self.rd, speed=self.speed, max_frames=self.max_frames, frame_skip=self.frame_skip,
                             seed=seed, num_of_apples=self.num_of_apples, human_control=self.human_control,
                             debug=self.is_debug)

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
            pygame.display.set_caption(FoodCollector.get_game_name())
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + int(self.tile_size/2)))
        else:
            self.screen = pygame.Surface((self.screen_size, self.screen_size + int(self.tile_size/2)))
        self.rc_manager = ResourceManager(current_path=self.current_path, font_size=self.font_size,
                                          tile_size=self.tile_size, is_render=self.rd)
        self.font = self.rc_manager.get_font()
        self.stage_map = StageMap(self.num_of_tiles, tile_size=self.tile_size, current_path=self.current_path,
                                  sprites=self.sprites, trees=self.trees, lands=self.lands, resources_manager=self.rc_manager)

    def __generate_players(self):
        pos_x = 0
        pos_y = 0

        self.player1 = CharacterSprite(self.tile_size, pos_x=pos_x, pos_y=pos_y,
                                       sprite_bg=((self.rc_manager.get_image(ResourceManager.GUY_L_1_TILE),self.rc_manager.get_image(ResourceManager.GUY_L_2_TILE)),
                                             (self.rc_manager.get_image(ResourceManager.GUY_R_1_TILE),self.rc_manager.get_image(ResourceManager.GUY_R_2_TILE)),
                                             (self.rc_manager.get_image(ResourceManager.GUY_U_1_TILE),self.rc_manager.get_image(ResourceManager.GUY_U_2_TILE)),
                                             (self.rc_manager.get_image(ResourceManager.GUY_D_1_TILE),self.rc_manager.get_image(ResourceManager.GUY_D_2_TILE))),
                                       speed=self.player_speed
                                       )
        # self.sprites.add(self.player1)
        self.players.add(self.player1)

    def __is_free_slot(self, x, y):
        for tree in self.trees:
            if x == tree.pos_x and y == tree.pos_y:
                return False

        for apple in self.apples:
            if x == apple.pos_x and y == apple.pos_y:
                return False

        for f in self.food:
            if x == f.pos_x and y == f.pos_y:
                return False

        for player in self.players:
            if x == player.pos_x and y == player.pos_y:
                return False

        for door in self.doors:
            if x == door.pos_x and y == door.pos_y:
                return False

        if x == 0 and y == 0:
            return False

        if x == self.num_of_tiles-1 and y == self.num_of_tiles-1:
            return False

        return True

    def __generate_food(self):

        # Add food
        for _ in range(self.num_of_apples):
            while True:
                x = np.random.randint(0, self.num_of_tiles - 1)
                y = np.random.randint(0, self.num_of_tiles - 1)

                if self.__is_free_slot(x, y):
                    apple = AppleSprite(self.tile_size, pos_x=x, pos_y=y,
                                        sprite_bg=self.rc_manager.get_image(ResourceManager.APPLE_TILE))
                    self.sprites.add(apple)
                    self.apples.add(apple)
                    break

        # Add rice
        for _ in range(self.num_of_apples):
            while True:
                x = np.random.randint(0, self.num_of_tiles - 1)
                y = np.random.randint(0, self.num_of_tiles - 1)

                if self.__is_free_slot(x, y):
                    key = FoodSprite(self.tile_size, pos_x=x, pos_y=y,
                                     sprite_bg=self.rc_manager.get_image(ResourceManager.FOOD_TILE))
                    self.sprites.add(key)
                    self.food.add(key)
                    break

    def __generate_particles(self):
        # Add door
        door = DoorSprite(self.tile_size, self.num_of_tiles-1, self.num_of_tiles-1,
                          self.rc_manager.get_image(ResourceManager.DOOR_TILE))
        self.sprites.add(door)
        self.doors.add(door)

    def __draw_score(self):
        total_score = self.font.render('Score:' + str(self.total_score) + ',' + str(self.total_score2), False, Utils.get_color(Utils.WHITE))
        self.screen.blit(total_score, (10, self.screen_size + total_score.get_height()/1.3))

        life = self.font.render('Energy:', False, Utils.get_color(Utils.WHITE))
        self.screen.blit(life, (self.screen_size/2 - life.get_width()/2, self.screen_size + total_score.get_height()/1.3))

    @staticmethod
    def __is_key_pressed():
        keys = pygame.key.get_pressed()
        for i in range(len(keys)):
            if keys[i] != 0:
                return i
        return -1

    def __human_control(self, key):
        if self.human_control:
            if key == pygame.K_LEFT:
                self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
            if key == pygame.K_RIGHT:
                self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
            if key == pygame.K_UP:
                self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
            if key == pygame.K_DOWN:
                self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
            if key == pygame.K_a:
                self.__eat()
            if key == pygame.K_s:
                self.__pick()

    def __handle_event(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.reset()
                sys.exit()

        if not self.human_control:
            return True

        key = FoodCollector.__is_key_pressed()
        if key >= 0:
            self.__human_control(key)

        return True

    @staticmethod
    def get_key_pressed():
        return FoodCollector.__is_key_pressed()

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
                print("Total eat:", self.total_score2)
                print("Frame speed (FPS):", self.frame_speed)
                print("")

    def __draw_life(self):
        if self.total_score < 50:
            self.life_fq = 10
        elif self.total_score < 100:
            self.life_fq = 9
        elif self.total_score < 150:
            self.life_fq = 8
        elif self.total_score < 200:
            self.life_fq = 7
        elif self.total_score < 250:
            self.life_fq = 6
        elif self.total_score < 300:
            self.life_fq = 5
        elif self.total_score < 350:
            self.life_fq = 4
        elif self.total_score < 400:
            self.life_fq = 3
        elif self.total_score < 450:
            self.life_fq = 2
        else:
            self.life_fq = 1
        if self.frames_count % self.life_fq == 0:
            self.life = self.life - 2 * GlobalConstants.LIVES_FACTOR_SPEED
        if self.life < 0:
            self.life = 0
            self.end_of_game = True
        max_width = self.screen_size/2.6
        current_width = max_width*self.life/100
        pygame.draw.rect(self.screen, Utils.get_color(Utils.WHITE), [self.screen_size/2 + self.tile_size/1.5,
                                                                     self.screen_size + self.tile_size/8,
                                                                     current_width,
                                                                     self.tile_size/4])

    def __check_goal(self):
        for shop in self.doors:
            if self.player1.pos_x == shop.pos_x and self.player1.pos_y == shop.pos_y:
                self.total_score = self.total_score + GlobalConstants.FIND_SHOP_REWARD
                self.rewards.append(GlobalConstants.FIND_SHOP_REWARD)
                self.end_of_game = True
                return

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

        # Draw lives
        self.__draw_life()

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

        for apple in self.apples:
            apple.kill()

        for door in self.doors:
            door.kill()

        for tree in self.trees:
            tree.kill()

        for food in self.food:
            food.kill()

        for land in self.lands:
            land.kill()

        self.stage_map.load_map(self.current_stage)

        # Create door
        self.__generate_particles()

        # Create food
        self.__generate_food()

        # Create players
        self.__generate_players()

        if self.is_debug:
            interval = Utils.get_current_time() - self.started_time
            print("#################  RESET GAME  ##################")
            print("Episode terminated after:", interval, "(s)")
            print("Total score:", self.total_score)
            print("Total eat:", self.total_score2)
            print("#################################################")

        self.total_score = 0
        self.total_score2 = 0
        self.life = 100

        self.__render()

    def __check_reward(self):
        if len(self.rewards) <= 0:
            r1 = 0
        else:
            r1 = self.rewards.pop()

        if len(self.rewards_eat) <= 0:
            r2 = 0
        else:
            r2 = self.rewards_eat.pop()

        return [r1, r2]

    def __pick(self):
        for food in self.food:
            if food.pos_x == self.player1.pos_x and food.pos_y == self.player1.pos_y:
                self.total_score = self.total_score + GlobalConstants.PICK_RICE_REWARD
                self.rewards.append(GlobalConstants.PICK_RICE_REWARD)
                self.food.remove(food)
                self.sprites.remove(food)
                food.kill()
                if len(self.apples) == 0 and len(self.food) == 0:
                    self.__generate_food()
                return

        for food in self.apples:
            if food.pos_x == self.player1.pos_x and food.pos_y == self.player1.pos_y:
                self.total_score = self.total_score + GlobalConstants.PICK_FOOD_REWARD
                self.rewards.append(GlobalConstants.PICK_FOOD_REWARD)
                self.apples.remove(food)
                self.sprites.remove(food)
                food.kill()
                if len(self.apples) == 0 and len(self.food) == 0:
                    self.__generate_food()
                return

    def __eat(self):
        for food in self.apples:
            if food.pos_x == self.player1.pos_x and food.pos_y == self.player1.pos_y:
                self.rewards_eat.append(GlobalConstants.EAT_REWARD)
                self.total_score2 = self.total_score2 + GlobalConstants.EAT_REWARD
                self.life = self.life + 50
                if self.life > 100:
                    self.life = 100
                self.apples.remove(food)
                self.sprites.remove(food)
                food.kill()
                if len(self.apples) == 0 and len(self.food) == 0:
                    self.__generate_food()
                return

    def step(self, action):
        if self.human_control:
            raise ValueError("Error: human control mode")
        if not self.human_control:
            if action == GlobalConstants.LEFT_ACTION:
                self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
            elif action == GlobalConstants.RIGHT_ACTION:
                self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
            elif action == GlobalConstants.UP_ACTION:
                self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
            elif action == GlobalConstants.DOWN_ACTION:
                self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
            elif action == GlobalConstants.PICK_ACTION:
                self.__pick()
            elif action == GlobalConstants.EAT_ACTION:
                self.__eat()

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
        return range(self.num_of_actions)

    def get_state(self):
        pygame.pixelcopy.surface_to_array(self.current_buffer, self.screen)
        return self.current_buffer

    def is_terminal(self):
        return self.end_of_game

    def debug(self):
        self.__print_info()

    def get_num_of_actions(self):
        return self.num_of_actions

    def is_render(self):
        return self.rd

    def get_num_of_agents(self):
        return 1


def check_map():
    from PIL import Image
    from pathlib import Path
    home = str(Path.home())
    game = FoodCollector(render=True, speed=60, max_frames=10000, frame_skip=1, seed=None,
                         num_of_apples=1, human_control=False, debug=True)
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
            # break


def check_control():
    game = FoodCollector(render=True, speed=30, max_frames=10000, frame_skip=1, seed=None,
                         num_of_apples=1, human_control=True, debug=True)
    num_of_actions = game.get_num_of_actions()
    game.reset()

    while True:
        #random_action = np.random.randint(0, num_of_actions)
        #reward = game.step(random_action)
        #next_state = Utils.process_state(game.get_state())
        game.render()
        is_terminal = game.is_terminal()

        if is_terminal:
            print("Total Score", game.total_score)
            game.reset()
            break


if __name__ == '__main__':
    # check_map()
    check_control()
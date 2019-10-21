from fruit.envs.games.deep_sea_treasure.mdp import TransitionList
from fruit.envs.games.deep_sea_treasure.sprites import SubmarineSprite, TreasureSprite
from fruit.envs.games.utils import Utils
import numpy as np
import pygame
import os
import sys


class DeepSeaTreasureConstants:
    LINEAR = 0
    CONCAVE = 1
    CONVEX = 2
    MIXED = 3

    RIGHT_ACTION = 0
    LEFT_ACTION = 1
    DOWN_ACTION = 2
    UP_ACTION = 3


class DeepSeaTreasure(object):
    def __init__(self, width, min_depth=3, min_vertical_step=1, max_vertical_step=3, transition_noise=0.,
                 reward_noise=0., front_shape=DeepSeaTreasureConstants.CONCAVE, seed=None, min_treasure=1,
                 max_treasure=1000, render=False, speed=60, agent_random_location=False,
                 reshape_reward_weights=None, is_debug=False, graphical_state=False):
        self.num_of_cols = width
        self.min_depth = min_depth
        self.min_vertical_step = min_vertical_step
        self.max_vertical_step = max_vertical_step
        self.min_treasure = min_treasure
        self.max_treasure = max_treasure
        self.depths = [min_depth] * self.num_of_cols
        self.steps = [min_depth] * self.num_of_cols
        self.agent_row = 0
        self.agent_col = 0
        self.total_steps = 0
        self.weights = reshape_reward_weights
        self.chose_weight = 0

        step_range = max_vertical_step - min_vertical_step + 1
        self.trs = [None for _ in range(width)]
        self.is_debug = is_debug
        self.log_freq = 200
        self.graphical_state = graphical_state
        self.total_score = 0
        self.total_score_2 = 0

        if seed is None or seed < 0 or seed >= 9999:
            if seed is not None and (seed < 0 or seed >= 9999):
                print("Invalid seed ! Default seed = randint(0, 9999")
            self.seed = np.random.randint(0, 9999)
            self.random_seed = True
        else:
            self.random_seed = False
            self.seed = seed
            np.random.seed(seed)

        self.agent_random = agent_random_location
        if self.agent_random:
            self.agent_col = np.random.randint(0, self.num_of_cols - 1)
        for i in range(self.num_of_cols):
            if i > 0:
                self.depths[i] = self.depths[i - 1] + np.random.randint(step_range - 1) + min_vertical_step
                self.steps[i] = i + self.depths[i]
        self.num_of_rows = self.depths[-1] + 1
        self.front_shape = front_shape
        self.treasures = self.__set_treasure(min_treasure, max_treasure)
        self.transition_noise = transition_noise
        self.reward_noise = reward_noise
        self.num_of_actions = 4
        self.transition_function = self.__set_transitions()
        self.pareto_solutions = self.__get_pareto_solutions()
        self.num_of_objectives = 2
        self.rd = render
        if self.num_of_cols > 60:
            if self.rd:
                print("Could not render when width > 60")
                self.rd = False
        self.screen = None
        self.speed = speed
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.width = 50 * self.num_of_cols
        self.height = int((self.width / self.num_of_cols) * self.num_of_rows)
        self.tile_width = int(self.width / self.num_of_cols)
        self.player_width = int(self.tile_width * 5 / 6)
        self.player_height = int(self.player_width * 55/76)
        self.sprites = pygame.sprite.Group()
        self.current_buffer = np.array([[[0, 0, 0] for _ in range(self.height)] for _ in range(self.width)])

        # Initialize
        self.__init_pygame_engine()

        self.__generate_submarine()

        self.__generate_treasures()

        # Render the first frame
        self.__render()

    def __get_transition(self, col, row, action):
        new_row = row
        new_col = col

        if row > self.depths[col]:
            return col * self.num_of_rows + row

        # Right
        if action == 0:
            new_col = new_col + 1
            if new_col > self.num_of_cols - 1:
                new_col = new_col - 1

        # Left
        if action == 1:
            new_col = new_col - 1
            if new_col < 0:
                new_col = 0
            elif not self.__is_valid(new_row, new_col):
                new_col = new_col + 1

        # Down
        if action == 2:
            new_row = new_row + 1
            if new_row > self.depths[col]:
                new_row = new_row - 1

        # Up
        if action == 3:
            new_row = new_row - 1
            if new_row < 0:
                new_row = 0

        return new_col * self.num_of_rows + new_row

    def get_depths(self):
        return self.depths

    def __set_treasure(self, min_treasure, max_treasure):
        treasures = [0] * self.num_of_cols
        treasures[0] = min_treasure
        treasures[-1] = max_treasure
        steps_range = self.steps[-1] - self.steps[0]
        treasure_range = max_treasure - min_treasure
        for i in range(self.num_of_cols - 1):
            if i > 0:
                ratio = (self.steps[i] - self.steps[0]) / steps_range

                adjustment = ratio * (1 - ratio)

                treasures[i] = int(ratio * treasure_range + min_treasure + 0.5)

                if self.front_shape == DeepSeaTreasureConstants.CONVEX:
                    treasures[i] = treasures[i] + adjustment * treasure_range
                elif self.front_shape == DeepSeaTreasureConstants.CONCAVE:
                    treasures[i] = treasures[i] - adjustment * treasure_range
                elif self.front_shape == DeepSeaTreasureConstants.MIXED:
                    treasures[i] = treasures[i] + adjustment * treasure_range * (np.random.uniform(0., 1.) * 2 - 1)

        for i in range(self.num_of_cols):
            self.steps[i] = -self.steps[i]

        return treasures

    def __is_valid(self, row, col):
        valid = True
        if row > self.depths[col]:
            valid = False
        return valid

    def get_pareto_solutions(self):
        return self.pareto_solutions

    def __get_pareto_solutions(self):
        solutions = []
        if self.transition_noise == 0.0:
            for i in range(self.num_of_cols):
                solutions.append([self.treasures[i], self.steps[i]])
            return solutions
        else:
            return None

    def get_treasure(self):
        return self.treasures

    def __set_transitions(self):
        prob_each_action = self.transition_noise / self.num_of_actions
        transition_function = [[0] * self.num_of_actions for _ in range(self.num_of_cols * self.num_of_rows)]
        next_state = [0] * self.num_of_actions
        for col in range(self.num_of_cols):
            for row in range(self.num_of_rows):
                state = col * self.num_of_rows + row
                for act in range(self.num_of_actions):
                    next_state[act] = self.__get_transition(col, row, act)
                for act in range(self.num_of_actions):
                    li = TransitionList()
                    li.add(next_state[act], 1 - self.transition_noise)
                    for act2 in range(self.num_of_actions):
                        li.add(next_state[act2], prob_each_action)
                    transition_function[state][act] = li
        return transition_function

    def __update_position(self, action):
        state = self.__get_state_index()
        next_state = self.transition_function[state][action].get_next_state()
        self.agent_row = next_state % self.num_of_rows
        self.agent_col = int(next_state / self.num_of_rows)
        offset_w = int((self.tile_width - self.player_width) / 2)
        offset_h = int((self.tile_width - self.player_height) / 2)
        pos_y = self.agent_row * self.tile_width + offset_h
        pos_x = self.agent_col * self.tile_width + offset_w
        self.player.rect.x = pos_x
        self.player.rect.y = pos_y

    def __draw_score(self):
        total_score = self.font.render(str(self.total_score) + " " + str(self.total_score_2),
                                       False, Utils.get_color(Utils.WHITE))
        self.screen.blit(total_score, (self.width/2 - total_score.get_width()/2,
                                       self.height - total_score.get_height()*2))

    @staticmethod
    def get_game_name():
        return "DEEP SEA TREASURE"

    def __check_reward(self):
        col = self.agent_col
        row = self.agent_row
        objs = [0., -1.]
        if row == self.depths[col]:
            objs[0] = self.treasures[col]
        self.total_score = int(self.total_score + objs[0])
        self.total_score_2 = int(self.total_score_2 + objs[1])
        if self.weights is not None:
            w = self.weights[self.chose_weight]
            return np.multiply(objs, w)
        else:
            return objs

    def clone(self):
        if self.random_seed:
            seed = np.random.randint(0, 9999)
        else:
            seed = self.seed
        return DeepSeaTreasure(width=self.num_of_cols, min_depth=self.min_depth,
                               min_vertical_step=self.min_vertical_step,
                               max_vertical_step=self.max_vertical_step, transition_noise=self.transition_noise,
                               reward_noise=self.reward_noise, front_shape=self.front_shape, seed=seed,
                               min_treasure=self.min_treasure, max_treasure=self.max_treasure, render=self.rd,
                               speed=self.speed, agent_random_location=self.agent_random,
                               reshape_reward_weights=self.weights, is_debug=self.is_debug,
                               graphical_state=self.graphical_state)

    def get_num_of_objectives(self):
        return self.num_of_objectives

    def get_seed(self):
        return self.seed

    def __init_pygame_engine(self):

        # Center the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Init Pygame engine
        pygame.init()

        self.font = pygame.font.Font(self.current_path + "/../../common/fonts/font.ttf", 12)

        if self.rd:
            pygame.display.set_caption(DeepSeaTreasure.get_game_name())
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height))

    def __generate_submarine(self):
        offset_w = int((self.tile_width - self.player_width) / 2)
        offset_h = int((self.tile_width - self.player_height) / 2)
        image = pygame.image.load(self.current_path + "/graphics/submarine.png")
        if self.rd:
            image = pygame.transform.scale(image, (self.player_width, self.player_height)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.player_width, self.player_height))

        pos_y = self.agent_row * self.tile_width + offset_h
        pos_x = self.agent_col * self.tile_width + offset_w
        self.player = SubmarineSprite(pos_x=pos_x, pos_y=pos_y, sprite_bg=image)
        self.sprites.add(self.player)

    def __generate_treasures(self):
        color_blue = (100, 100, 100)
        color_black = (0, 0, 0)
        offset = int((self.tile_width - self.player_width) / 2)
        image = pygame.image.load(self.current_path + "/graphics/treasure.png")
        if self.rd:
            image = pygame.transform.scale(image, (self.player_width, self.player_width)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.player_width, self.player_width))

        for row in range(self.num_of_rows):
            for col in range(self.num_of_cols):
                if row <= self.depths[col]:
                    pygame.draw.rect(self.screen, color_blue,
                                     pygame.Rect(col * self.tile_width , row * self.tile_width,
                                                 self.tile_width - 1, self.tile_width - 1))
                    if row == self.depths[col]:
                        self.trs[col] = TreasureSprite(pos_x=col * self.tile_width + offset,
                                                       pos_y=row * self.tile_width + offset, sprite_bg=image)
                        self.sprites.add(self.trs[col])
                else:
                    pygame.draw.rect(self.screen, color_black,
                                     pygame.Rect(col * self.tile_width, row * self.tile_width,
                                                 self.tile_width, self.tile_width))

    @staticmethod
    def __is_key_pressed():
        keys = pygame.key.get_pressed()
        for i in range(len(keys)):
            if keys[i] != 0:
                return i
        return -1

    def __human_control(self, key):
        action = -1
        if key == pygame.K_LEFT:
            action = DeepSeaTreasureConstants.LEFT_ACTION
        if key == pygame.K_RIGHT:
            action = DeepSeaTreasureConstants.RIGHT_ACTION
        if key == pygame.K_UP:
            action = DeepSeaTreasureConstants.UP_ACTION
        if key == pygame.K_DOWN:
            action = DeepSeaTreasureConstants.DOWN_ACTION
        return action

    def __handle_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        key = DeepSeaTreasure.__is_key_pressed()
        if key >= 0:
            return self.__human_control(key)

        return -1

    def get_key_pressed(self):
        return DeepSeaTreasure.__is_key_pressed()

    def __print_info(self):
        if self.is_debug:
            if self.total_steps % self.log_freq == 0:
                print("############# STATE ###############")
                for row in range(self.num_of_rows):
                    line = ""
                    for col in range(self.num_of_cols):
                        state = col * self.num_of_rows + row
                        line = line + str(state)
                        if row == self.agent_row and col == self.agent_col:
                            line = line + "*\t\t"
                        else:
                            if row >= self.depths[col]:
                                line = line + "x\t\t"
                            else:
                                line = line + "-\t\t"
                    print(line)
                sep = ''
                for i in range(self.num_of_cols):
                    sep = sep + "-------"
                print(sep)
                score = ""
                for i in range(self.num_of_cols):
                    score = score + str(int(self.treasures[i])) + "\t\t"
                print(score)
                print("###################################\n")
                print("\n##### Pareto Solutions #####")
                if self.pareto_solutions is not None:
                    for i in range(self.num_of_cols):
                        print(self.pareto_solutions[i])

    def __render(self, is_agent=False):
        human_action = -1
        if self.rd:
            human_action = self.__handle_event()
        if not is_agent:
            if human_action != -1:
                self.__update_position(human_action)

        self.__generate_treasures()

        self.__draw_score()

        # Update sprites
        self.sprites.update()

        # Redraw all sprites
        self.sprites.draw(self.screen)

        # Show to the screen what we're have drawn so far
        if self.rd:
            pygame.display.flip()

        # Debug
        self.__print_info()

        # Maintain 20 fps
        pygame.time.Clock().tick(self.speed)

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):

        for sprite in self.sprites:
            sprite.kill()

        self.agent_col = self.agent_row = 0
        if self.agent_random:
            self.agent_row = 0
            self.agent_col = np.random.randint(0, self.num_of_cols+1)

        self.__generate_submarine()

        self.__render()

        self.total_score_2 = self.total_score = 0

    def step(self, action):

        self.__update_position(action)
        self.total_steps = self.total_steps + 1
        self.__render(True)

        return self.__check_reward()

    def render(self, is_agent=False):
        self.__render(is_agent)

    def step_all(self, action):
        rewards = self.step(action)
        next_state = self.get_state()
        terminal = self.is_terminal()
        return next_state, rewards, terminal, 0

    def get_state_space(self):
        if self.graphical_state:
            return [self.width, self.height]
        else:
            from fruit.types.priv import Space
            return Space(0, self.num_of_cols * self.num_of_rows - 1, True)

    def get_action_space(self):
        return range(self.num_of_actions)

    def __get_state_index(self):
        return self.agent_col * self.num_of_rows + self.agent_row

    def get_state(self):
        if self.graphical_state:
            pygame.pixelcopy.surface_to_array(self.current_buffer, self.screen)
            return self.current_buffer
        else:
            return self.agent_col * self.num_of_rows + self.agent_row

    def is_terminal(self):
        # print(self.total_steps, self.max_steps)
        # if self.total_steps > self.max_steps:
        #    return True
        if self.agent_row == self.depths[self.agent_col]:
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
    game = DeepSeaTreasure(width=5, render=True, graphical_state=True, speed=10)
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
        #count = count + 1
        #img = Image.fromarray(next_state, 'L')
        #img.save(full_path + str(count) +'.png')

        if is_terminal:
            print("Total Score", game.total_score)
            game.reset()
            # break


if __name__ == '__main__':
    check_map()
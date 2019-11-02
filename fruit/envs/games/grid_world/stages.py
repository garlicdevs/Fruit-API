import random
import numpy as np
from fruit.envs.games.grid_world.constants import GlobalConstants
from fruit.envs.games.grid_world.manager import ResourceManager
from fruit.envs.games.grid_world.sprites import LandSprite, KeySprite, PlantSprite, MinusSprite


class StageMap(object):
    def __init__(self, num_of_rows, num_of_columns, tile_size, current_path, seed, num_of_obstacles,
                 sprites, obstacles, lands, keys, minuses, resources_manager):
        self.num_of_stages = 2
        self.num_of_rows = num_of_rows
        self.num_of_columns = num_of_columns
        self.map = [None] * self.num_of_stages
        self.current_path = current_path
        self.sprites = sprites
        self.obstacles = obstacles
        self.tile_size = tile_size
        self.rc = resources_manager
        self.lands = lands
        self.seed = seed
        self.keys = keys
        self.minuses = minuses
        self.num_of_obstacles = num_of_obstacles

        if seed is not None:
            random.seed(seed)

        self.__build_map()

    def __build_map(self):

        #########################################################################
        #########################################################################
        # STAGE 1
        # Create a dynamic graph

        obs = random.sample(range(1, self.num_of_rows * self.num_of_columns - 1), self.num_of_obstacles)
        self.map[0] = [[-1 for _ in range(self.num_of_columns)] for _ in range(self.num_of_rows)]
        self.map[0][-1][-1] = GlobalConstants.KEY_TILE
        for o in obs:
            r = int(o/self.num_of_columns)
            c = int(o % self.num_of_rows)
            self.map[0][r][c] = GlobalConstants.PLANT_TILE
        #########################################################################

        #########################################################################
        # STAGE 2
        # Create a static graph
        self.map[1] = [[-1, -1,  1,  2, -1, -1, -1, -1, -1],
                       [-1,  0,  0,  0,  0,  0,  0,  0, -1],
                       [-1,  0, -1, -1, -1, -1, -1,  0, -1],
                       [-1,  0, -1, -1, -1, -1, -1,  0, -1],
                       [-1,  0,  0,  0, -1, -1, -1,  0, -1],
                       [-1, -1, -1, -1, -1, -1, -1,  0, -1],
                       [-1, -1,  0,  0,  0,  0,  0,  0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1]]

        print("==========================STAGE MAP==============================")
        print("GRID WORLD ({} x {})".format(self.num_of_columns, self.num_of_rows))
        print(np.array(self.map[1]))
        print("=================================================================")

    def load_map(self, stage):

        if stage >= self.num_of_stages:
            raise ValueError("Stage out of range !!!")

        for row in range(len(self.map[stage])):
            for col in range(len(self.map[stage][row])):
                land_bg = self.rc.get_image(ResourceManager.LAND_TILE)
                land = LandSprite(self.tile_size, col, row, land_bg)
                self.lands.add(land)

                if self.map[stage][row][col] == GlobalConstants.KEY_TILE:
                    key_bg = self.rc.get_image(ResourceManager.KEY_TILE)
                    key = KeySprite(self.tile_size, col, row, key_bg)
                    self.sprites.add(key)
                    self.keys.add(key)

                if self.map[stage][row][col] == GlobalConstants.PLANT_TILE:
                    plant_bg = self.rc.get_image(ResourceManager.PLANT_TILE)
                    plant = PlantSprite(self.tile_size, col, row, plant_bg)
                    self.sprites.add(plant)
                    self.obstacles.add(plant)

                if self.map[stage][row][col] == GlobalConstants.MINUS_TILE:
                    minus_bg = self.rc.get_image(ResourceManager.MINUS_TILE)
                    minus = MinusSprite(self.tile_size, col, row, minus_bg)
                    self.sprites.add(minus)
                    self.minuses.add(minus)

    def number_of_stages(self):
        return self.num_of_stages
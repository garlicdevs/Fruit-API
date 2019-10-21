from fruit.envs.games.tank_battle.constants import GlobalConstants
from fruit.envs.games.tank_battle.manager import ResourceManager
from fruit.envs.games.tank_battle.sprites import WallSprite


class StageMap(object):
    def __init__(self, num_of_tiles, tile_size, current_path, sprites, walls, resources_manager):
        self.num_of_stages = 1
        self.num_of_tiles = num_of_tiles
        self.map = [None] * self.num_of_stages
        self.current_path = current_path
        self.sprites = sprites
        self.walls = walls
        self.tile_size = tile_size
        self.rc = resources_manager

        self.__build_map()

    def __build_map(self):
        #########################################################################
        #########################################################################
        # STAGE 1
        # We can make a static or dynamic map
        # This is a static map (it is better to use dynamic when num_of_tiles is
        # unknown). However, static map is easier to create a stage.
        self.map[0] = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1,  0,  0,  0,  0,  2,  0,  0,  0,  0, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

        # This is an example of dynamic map
        # center_y = int(self.num_of_tiles/2)
        # wall_positions = []
        # for i in range(2, self.num_of_tiles-2):
        #     wall_positions.append([i, center_y, GlobalConstants.SOFT_OBJECT])
        # self.map[0] = wall_positions
        # END OF STAGE 1
        #########################################################################
        #########################################################################

    def load_map(self, stage):
        if stage >= self.num_of_stages:
            raise ValueError("Stage out of range !!!")
        # This is for dynamic map
        # wall_bg = self.rc.get_image(ResourceManager.SOFT_WALL)
        # for pos in self.map[stage]:
        #     wall = WallSprite(self.tile_size, pos[0], pos[1], wall_bg)
        #     wall.type = pos[2]
        #     self.sprites.add(wall)
        #     self.walls.add(wall)

        # This is for static map
        for row in range(len(self.map[stage])):
            for col in range(len(self.map[stage][row])):
                if self.map[stage][row][col] == GlobalConstants.WALL_TILE:
                    wall_bg = self.rc.get_image(ResourceManager.SOFT_WALL)
                    wall = WallSprite(self.tile_size, col, row, wall_bg)
                    wall.type = GlobalConstants.SOFT_OBJECT
                    self.sprites.add(wall)
                    self.walls.add(wall)
                elif self.map[stage][row][col] == GlobalConstants.ROCK_TILE:
                    wall_bg = self.rc.get_image(ResourceManager.HARD_WALL)
                    wall = WallSprite(self.tile_size, col, row, wall_bg)
                    wall.type = GlobalConstants.HARD_OBJECT
                    self.sprites.add(wall)
                    self.walls.add(wall)
                elif self.map[stage][row][col] == GlobalConstants.SEA_TILE:
                    wall_bg = self.rc.get_image(ResourceManager.SEA_WALL)
                    wall = WallSprite(self.tile_size, col, row, wall_bg)
                    wall.type = GlobalConstants.TRANSPARENT_OBJECT
                    self.sprites.add(wall)
                    self.walls.add(wall)

    def number_of_stages(self):
        return self.num_of_stages
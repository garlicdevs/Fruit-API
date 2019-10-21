from fruit.envs.games.milk_factory.constants import GlobalConstants
from fruit.envs.games.milk_factory.manager import ResourceManager
from fruit.envs.games.milk_factory.sprites import LandSprite, TableSprite, ExitSprite


class StageMap(object):
    def __init__(self, num_of_tiles, tile_size, current_path, sprites, resources_manager):
        self.num_of_stages = 2
        self.num_of_tiles = num_of_tiles
        self.map = [None] * self.num_of_stages
        self.current_path = current_path
        self.sprites = sprites
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
        self.map[0] = [[ 0,  0,  0,  0,  0],
                       [-1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1],
                       [-1, -1,  1, -1, -1]]

        self.map[1] = [[ 0,  0,  0,  0,  0],
                       [-1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1],
                       [ 1, -1, -1, -1,  1]]

        #########################################################################
        #########################################################################

    def load_map(self, stage, conveyor_sprites, target_sprites):
        if stage >= self.num_of_stages:
            raise ValueError("Stage out of range !!!")

        # This is for static map
        for row in range(len(self.map[stage])):
            for col in range(len(self.map[stage][row])):
                land_bg = self.rc.get_image(ResourceManager.LAND_TILE)
                land = LandSprite(self.tile_size, col, row, land_bg)
                self.sprites.add(land)

                if self.map[stage][row][col] == GlobalConstants.TABLE_TILE:
                    table_bg = self.rc.get_image(ResourceManager.TABLE_TILE)
                    table = TableSprite(self.tile_size, col, row, table_bg)
                    conveyor_sprites.add(table)
                    self.sprites.add(table)

                if self.map[stage][row][col] == GlobalConstants.EXIT_TILE:
                    exit_sprite = ExitSprite(self.tile_size, col, row, self.rc.get_image(ResourceManager.EXIT_TILE))
                    target_sprites.add(exit_sprite)
                    self.sprites.add(exit_sprite)

    def number_of_stages(self):
        return self.num_of_stages
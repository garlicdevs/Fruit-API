from fruit.envs.games.food_collector_.constants import GlobalConstants
from fruit.envs.games.food_collector_.manager import ResourceManager
from fruit.envs.games.food_collector_.sprites import TreeSprite, LandSprite


class StageMap(object):
    def __init__(self, num_of_tiles, tile_size, current_path, sprites, trees, lands, resources_manager):
        self.num_of_stages = 1
        self.num_of_tiles = num_of_tiles
        self.map = [None] * self.num_of_stages
        self.current_path = current_path
        self.sprites = sprites
        self.trees = trees
        self.tile_size = tile_size
        self.rc = resources_manager
        self.lands = lands

        self.__build_map()

    def __build_map(self):
        #########################################################################
        #########################################################################
        # STAGE 1
        # We can make a static or dynamic map
        # This is a static map (it is better to use dynamic when num_of_tiles is
        # unknown). However, static map is easier to create a stage.
        self.map[0] = [[-1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1],
                       [ 1, -1,  1, -1, -1, -1],
                       [-1, -1, -1,  1, -1, -1],
                       [-1,  0, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1]]

        #########################################################################
        #########################################################################

    def load_map(self, stage):
        if stage >= self.num_of_stages:
            raise ValueError("Stage out of range !!!")

        # This is for static map
        for row in range(len(self.map[stage])):
            for col in range(len(self.map[stage][row])):
                land_bg = self.rc.get_image(ResourceManager.LAND_TILE)
                land = LandSprite(self.tile_size, col, row, land_bg)
                self.lands.add(land)

                if self.map[stage][row][col] == GlobalConstants.TREE_TILE:
                    tree_bg = self.rc.get_image(ResourceManager.TREE_TILE)
                    tree = TreeSprite(self.tile_size, col, row, tree_bg)
                    self.sprites.add(tree)
                    self.trees.add(tree)

                if self.map[stage][row][col] == GlobalConstants.PLANT_TILE:
                    plant_bg = self.rc.get_image(ResourceManager.PLANT_TILE)
                    plant = TreeSprite(self.tile_size, col, row, plant_bg)
                    self.sprites.add(plant)
                    self.trees.add(plant)

    def number_of_stages(self):
        return self.num_of_stages
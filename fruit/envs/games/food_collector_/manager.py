import pygame
from fruit.envs.games.utils import Utils


class ResourceManager(object):
    APPLE_TILE = "apple.png"
    DOOR_TILE = "door.png"
    FOOD_TILE = "gold.png"
    TREE_TILE = "tree.png"
    LAND_TILE = "land.png"
    PLANT_TILE = "plant.png"
    GUY_L_1_TILE = "guy_l_1.png"
    GUY_L_2_TILE = "guy_l_2.png"
    GUY_R_1_TILE = "guy_r_1.png"
    GUY_R_2_TILE = "guy_r_2.png"
    GUY_U_1_TILE = "guy_u_1.png"
    GUY_U_2_TILE = "guy_u_2.png"
    GUY_D_1_TILE = "guy_d_1.png"
    GUY_D_2_TILE = "guy_d_2.png"

    def __init__(self, current_path, font_size, tile_size, is_render):
        self.font_size = font_size
        self.tile_size = tile_size
        self.bullet_size = int(tile_size/6)
        self.current_path = current_path + '/graphics/'
        self.render = is_render
        pygame.font.init()
        self.font = pygame.font.Font(self.current_path + "../../../common/fonts/font.ttf", self.font_size)
        self.font.set_bold(True)
        self.resources = {}
        self.__add_resources()

    def __add_resources(self):
        image = pygame.image.load(self.current_path + ResourceManager.APPLE_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.APPLE_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.PLANT_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.PLANT_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.LAND_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.LAND_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.DOOR_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.DOOR_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.FOOD_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.FOOD_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.TREE_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.TREE_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_L_1_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_L_1_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_L_2_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_L_2_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_R_1_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_R_1_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_R_2_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_R_2_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_U_1_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_U_1_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_U_2_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_U_2_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_D_1_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_D_1_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.GUY_D_2_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.GUY_D_2_TILE] = image

    def get_image(self, key):
        return self.resources[key]

    def get_font(self):
        return self.font
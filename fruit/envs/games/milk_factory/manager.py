import pygame


class ResourceManager(object):
    ROBOT_MILK = "robot_1.png"
    ROBOT_MILK_ERROR = 'robot_1_error.png'
    ROBOT_MILK_PICKED = 'robot_1_picked.png'
    ROBOT_FIX = "robot_2.png"
    LAND_TILE = "land.png"
    MILK_TILE = "milk.png"
    TABLE_TILE = "sea.png"
    EXIT_TILE = "exit.png"
    ERROR_TILE = 'error.png'

    def __init__(self, current_path, font_size, tile_size, is_render):
        self.font_size = font_size
        self.tile_size = tile_size
        self.current_path = current_path + '/graphics/'
        self.render = is_render
        pygame.font.init()
        self.font = pygame.font.Font(self.current_path + "../../../common/fonts/font.ttf", self.font_size)
        self.font.set_bold(True)
        self.resources = {}
        self.__add_resources()

    def __add_resources(self):
        image = pygame.image.load(self.current_path + ResourceManager.ROBOT_MILK)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.ROBOT_MILK] = image

        image = pygame.image.load(self.current_path + ResourceManager.ROBOT_MILK_ERROR)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.ROBOT_MILK_ERROR] = image

        image = pygame.image.load(self.current_path + ResourceManager.ROBOT_MILK_PICKED)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.ROBOT_MILK_PICKED] = image

        image = pygame.image.load(self.current_path + ResourceManager.ROBOT_FIX)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.ROBOT_FIX] = image

        image = pygame.image.load(self.current_path + ResourceManager.LAND_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.LAND_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.EXIT_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.EXIT_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.MILK_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.MILK_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.ERROR_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.ERROR_TILE] = image

        image = pygame.image.load(self.current_path + ResourceManager.TABLE_TILE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.TABLE_TILE] = image

    def get_image(self, key):
        return self.resources[key]

    def get_font(self):
        return self.font
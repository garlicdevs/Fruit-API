import pygame
from fruit.envs.games.utils import Utils


class ResourceManager(object):
    HARD_WALL = "rock.png"
    SOFT_WALL = "wall.png"
    PLAYER1_UP = "player1.png"
    PLAYER2_UP = "player5.png"
    BASE = "base_close_full.png"
    ENEMY_UP = "enemy.png"
    SEA_WALL = "sea.png"
    EXPLOSION_1 = "boom-large1.png"
    EXPLOSION_2 = "boom-large2.png"
    EXPLOSION_3 = "boom-large3.png"

    BULLET = "bullet"
    PLAYER1_LEFT = "player1_left"
    PLAYER1_RIGHT = "player1_right"
    PLAYER1_DOWN = "player1_down"
    PLAYER2_LEFT = "player2_left"
    PLAYER2_RIGHT = "player2_right"
    PLAYER2_DOWN = "player2_down"
    ENEMY_LEFT = "enemy_left"
    ENEMY_RIGHT = "enemy_right"
    ENEMY_DOWN = "enemy_down"

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
        image = pygame.image.load(self.current_path + ResourceManager.HARD_WALL)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))

        self.resources[ResourceManager.HARD_WALL] = image

        image = pygame.image.load(self.current_path + ResourceManager.SOFT_WALL)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.SOFT_WALL] = image

        image = pygame.image.load(self.current_path + ResourceManager.SEA_WALL)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.SEA_WALL] = image

        image = pygame.image.load(self.current_path + ResourceManager.BASE)
        if self.render:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
        self.resources[ResourceManager.BASE] = image

        image_up = pygame.image.load(self.current_path + ResourceManager.PLAYER1_UP)
        image_left = pygame.transform.rotate(image_up, 90)
        image_right = pygame.transform.rotate(image_up, -90)
        image_down = pygame.transform.rotate(image_up, 180)
        if self.render:
            image_up = pygame.transform.scale(image_up, (self.tile_size-1, self.tile_size-1)).convert_alpha()
            image_left = pygame.transform.scale(image_left, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_right = pygame.transform.scale(image_right, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_down = pygame.transform.scale(image_down, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
        else:
            image_up = pygame.transform.scale(image_up, (self.tile_size - 1, self.tile_size - 1))
            image_left = pygame.transform.scale(image_left, (self.tile_size - 1, self.tile_size - 1))
            image_right = pygame.transform.scale(image_right, (self.tile_size - 1, self.tile_size - 1))
            image_down = pygame.transform.scale(image_down, (self.tile_size - 1, self.tile_size - 1))
        self.resources[ResourceManager.PLAYER1_UP] = image_up
        self.resources[ResourceManager.PLAYER1_LEFT] = image_left
        self.resources[ResourceManager.PLAYER1_RIGHT] = image_right
        self.resources[ResourceManager.PLAYER1_DOWN] = image_down

        image_up = pygame.image.load(self.current_path + ResourceManager.PLAYER2_UP)
        image_left = pygame.transform.rotate(image_up, 90)
        image_right = pygame.transform.rotate(image_up, -90)
        image_down = pygame.transform.rotate(image_up, 180)
        if self.render:
            image_up = pygame.transform.scale(image_up, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_left = pygame.transform.scale(image_left, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_right = pygame.transform.scale(image_right, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_down = pygame.transform.scale(image_down, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
        else:
            image_up = pygame.transform.scale(image_up, (self.tile_size - 1, self.tile_size - 1))
            image_left = pygame.transform.scale(image_left, (self.tile_size - 1, self.tile_size - 1))
            image_right = pygame.transform.scale(image_right, (self.tile_size - 1, self.tile_size - 1))
            image_down = pygame.transform.scale(image_down, (self.tile_size - 1, self.tile_size - 1))
        self.resources[ResourceManager.PLAYER2_UP] = image_up
        self.resources[ResourceManager.PLAYER2_LEFT] = image_left
        self.resources[ResourceManager.PLAYER2_RIGHT] = image_right
        self.resources[ResourceManager.PLAYER2_DOWN] = image_down

        image_up = pygame.image.load(self.current_path + ResourceManager.ENEMY_UP)
        image_left = pygame.transform.rotate(image_up, 90)
        image_right = pygame.transform.rotate(image_up, -90)
        image_down = pygame.transform.rotate(image_up, 180)
        if self.render:
            image_up = pygame.transform.scale(image_up, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_left = pygame.transform.scale(image_left, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_right = pygame.transform.scale(image_right, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
            image_down = pygame.transform.scale(image_down, (self.tile_size - 1, self.tile_size - 1)).convert_alpha()
        else:
            image_up = pygame.transform.scale(image_up, (self.tile_size - 1, self.tile_size - 1))
            image_left = pygame.transform.scale(image_left, (self.tile_size - 1, self.tile_size - 1))
            image_right = pygame.transform.scale(image_right, (self.tile_size - 1, self.tile_size - 1))
            image_down = pygame.transform.scale(image_down, (self.tile_size - 1, self.tile_size - 1))
        self.resources[ResourceManager.ENEMY_UP] = image_up
        self.resources[ResourceManager.ENEMY_LEFT] = image_left
        self.resources[ResourceManager.ENEMY_RIGHT] = image_right
        self.resources[ResourceManager.ENEMY_DOWN] = image_down

        image = pygame.Surface([self.bullet_size, self.bullet_size])
        image.fill(Utils.get_color(Utils.WHITE))
        self.resources[ResourceManager.BULLET] = image

        image_1 = pygame.image.load(self.current_path + ResourceManager.EXPLOSION_1)
        image_2 = pygame.image.load(self.current_path + ResourceManager.EXPLOSION_2)
        image_3 = pygame.image.load(self.current_path + ResourceManager.EXPLOSION_3)
        if self.render:
            image_1 = pygame.transform.scale(image_1, (self.tile_size, self.tile_size)).convert_alpha()
            image_2 = pygame.transform.scale(image_2, (self.tile_size, self.tile_size)).convert_alpha()
            image_3 = pygame.transform.scale(image_3, (self.tile_size, self.tile_size)).convert_alpha()
        else:
            image_1 = pygame.transform.scale(image_1, (self.tile_size, self.tile_size))
            image_2 = pygame.transform.scale(image_2, (self.tile_size, self.tile_size))
            image_3 = pygame.transform.scale(image_3, (self.tile_size, self.tile_size))

        self.resources[ResourceManager.EXPLOSION_1] = image_1
        self.resources[ResourceManager.EXPLOSION_2] = image_2
        self.resources[ResourceManager.EXPLOSION_3] = image_3

    def get_image(self, key):
        return self.resources[key]

    def get_font(self):
        return self.font
import pygame
import numpy as np

from fruit.envs.games.grid_world.constants import GlobalConstants


class KeySprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.KEY_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class LandSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.LAND_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class PlantSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.PLANT_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class MinusSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.MINUS_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class CharacterSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg, num_of_rows, num_of_columns):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = -1

        self.direction = np.random.randint(0, 4)  # current direction
        self.num_of_rows = num_of_rows
        self.num_of_columns = num_of_columns

        self.image = sprite_bg[self.direction][1]
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size
        self.sprite_bg = sprite_bg

    def update(self):
        self.image = self.sprite_bg[self.direction][1]
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

    def move(self, action, rigid_objs):
        if action < 0:
            return True

        current_x = self.pos_x
        current_y = self.pos_y

        self.direction = action

        if action == GlobalConstants.LEFT_ACTION:
            current_x = current_x - 1
        elif action == GlobalConstants.RIGHT_ACTION:
            current_x = current_x + 1
        elif action == GlobalConstants.UP_ACTION:
            current_y = current_y - 1
        else:
            current_y = current_y + 1

        # Check if there is a obstacle at (current_x, current_y)
        can_move = True
        for obj in rigid_objs:
            if obj.type == GlobalConstants.PLANT_TILE:
                if current_x == obj.pos_x and current_y == obj.pos_y:
                    can_move = False
                    break

        # Check borders
        if current_x < 0 or current_y < 0 or current_x > self.num_of_columns - 1 or current_y > self.num_of_rows - 1:
            can_move = False

        if can_move:
            self.pos_x = current_x
            self.pos_y = current_y

        return can_move

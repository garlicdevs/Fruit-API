from fruit.envs.games.food_collector.constants import GlobalConstants
import pygame
import numpy as np


class AppleSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.APPLE_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class DoorSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.DOOR_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class FoodSprite(pygame.sprite.Sprite):
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


class TreeSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.TREE_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class CharacterSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg, speed):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = -1
        self.speed = speed
        self.direction = np.random.randint(0, 4)  # current direction

        self.image = sprite_bg[self.direction][1]
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size
        self.target_x = self.pos_x
        self.target_y = self.pos_y
        self.sprite_bg = sprite_bg

        self.animation = 1
        self.total_frames = 0

    def update(self):
        self.image = self.sprite_bg[self.direction][self.animation]
        self.total_frames = self.total_frames + 1
        if self.total_frames % int(self.speed/2) == 0:
            self.animation = self.animation + 1
            self.animation = self.animation % 2
            self.total_frames = 0
        if self.target_x != self.pos_x:
            dist = self.target_x - self.pos_x
            self.rect.x = self.rect.x + dist * self.speed
            if self.rect.x == self.target_x * self.size:
                self.pos_x = self.target_x
        if self.target_y != self.pos_y:
            dist = self.target_y - self.pos_y
            self.rect.y = self.rect.y + dist * self.speed
            if self.rect.y == self.target_y * self.size:
                self.pos_y = self.target_y

    def move(self, action, rigid_objs):
        if action < 0:
            return True

        # Wait the animation
        if self.target_x != self.pos_x or self.target_y != self.pos_y:
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
            if obj.type == GlobalConstants.TREE_TILE:
                if current_x == obj.pos_x and current_y == obj.pos_y:
                    can_move = False
                    break

        # Check borders
        num_of_tiles = int(GlobalConstants.SCREEN_SIZE/GlobalConstants.TILE_SIZE)
        if current_x < 0 or current_y < 0 or current_x > num_of_tiles - 1 or current_y > num_of_tiles - 1:
            can_move = False

        if can_move:
            self.target_x = current_x
            self.target_y = current_y

        return can_move

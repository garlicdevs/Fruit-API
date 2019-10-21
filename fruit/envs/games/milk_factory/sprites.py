import pygame
import numpy as np
from fruit.envs.games.milk_factory.constants import GlobalConstants


class MilkRobotSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, error_freq, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.ROBOT_TYPE_MILK

        self.sprite_bg = sprite_bg

        self.image = sprite_bg[0]
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

        self.error_freq = error_freq
        self.error = False
        self.count = 0

        self.picked = False
        self.num_of_picks = 0

    def redraw(self):
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

    def is_free_slot(self, x, y, sprites):
        for sprite in sprites:
            if type(sprite) is not LandSprite and type(sprite) is not ExitSprite:
                if sprite.pos_x == x and sprite.pos_y == y:
                    return False
        return True

    def fix(self):
        self.error = False
        self.count = 0
        # self.image = self.sprite_bg[0]
        # self.redraw()

    def unpick(self):
        self.picked = False
        # self.image = self.sprite_bg[0]
        # self.redraw()

    def pick(self):
        self.picked = True
        # self.image = self.sprite_bg[2]
        # self.redraw()
        self.num_of_picks += 1

    def update(self):
        rand = np.random.randint(0, 1000)
        if rand < self.error_freq * 1000:
            self.error = True
            # self.image = self.sprite_bg[1]
            # self.redraw()

    def move(self, action, sprites):
        if self.error:
            return 0
        num_of_rows = int(GlobalConstants.SCREEN_SIZE/GlobalConstants.TILE_SIZE)
        if action == GlobalConstants.LEFT_ACTION:
            if self.pos_x == 0:
                return 0
            else:
                if self.is_free_slot(self.pos_x-1, self.pos_y, sprites):
                    self.pos_x -= 1
        elif action == GlobalConstants.RIGHT_ACTION:
            if self.pos_x == num_of_rows - 1:
                return 0
            else:
                if self.is_free_slot(self.pos_x + 1, self.pos_y, sprites):
                    self.pos_x += 1
        elif action == GlobalConstants.UP_ACTION:
            if self.pos_y == 1:
                return 0
            else:
                if self.is_free_slot(self.pos_x, self.pos_y - 1, sprites):
                    self.pos_y -= 1
        elif action == GlobalConstants.DOWN_ACTION:
            if self.pos_y == num_of_rows - 1:
                return 0
            else:
                if self.is_free_slot(self.pos_x, self.pos_y + 1, sprites):
                    self.pos_y += 1
        elif action == GlobalConstants.FIRE_ACTION:
            for sprite in sprites:
                if type(sprite) is MilkSprite:
                    if sprite.pos_x == self.pos_x and sprite.pos_y + 1 == self.pos_y and not self.picked:
                        sprite.reset()
                        self.pick()
                        return 10
                elif type(sprite) is ExitSprite:
                    if sprite.pos_x == self.pos_x and sprite.pos_y == self.pos_y and self.picked:
                        self.unpick()
                        return 10

        self.redraw()
        return 0


class FixRobotSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.ROBOT_TYPE_FIX

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

    def redraw(self):
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

    def is_free_slot(self, x, y, sprites):
        for sprite in sprites:
            if type(sprite) is not LandSprite and type(sprite) is not ExitSprite:
                if sprite.pos_x == x and sprite.pos_y == y:
                    return False
        return True

    def move(self, action, sprites):
        num_of_rows = int(GlobalConstants.SCREEN_SIZE/GlobalConstants.TILE_SIZE)
        if action == GlobalConstants.LEFT_ACTION:
            if self.pos_x == 0:
                return 0
            else:
                if self.is_free_slot(self.pos_x-1, self.pos_y, sprites):
                    self.pos_x -= 1
        elif action == GlobalConstants.RIGHT_ACTION:
            if self.pos_x == num_of_rows - 1:
                return 0
            else:
                if self.is_free_slot(self.pos_x + 1, self.pos_y, sprites):
                    self.pos_x += 1
        elif action == GlobalConstants.UP_ACTION:
            if self.pos_y == 1:
                return 0
            else:
                if self.is_free_slot(self.pos_x, self.pos_y - 1, sprites):
                    self.pos_y -= 1
        elif action == GlobalConstants.DOWN_ACTION:
            if self.pos_y == num_of_rows - 1:
                return 0
            else:
                if self.is_free_slot(self.pos_x, self.pos_y + 1, sprites):
                    self.pos_y += 1
        elif action == GlobalConstants.FIRE_ACTION:
            for sprite in sprites:
                if type(sprite) is MilkRobotSprite:
                    if (sprite.pos_x == self.pos_x and sprite.pos_y + 1 == self.pos_y) or \
                            (sprite.pos_x == self.pos_x and sprite.pos_y - 1 == self.pos_y) or \
                            (sprite.pos_x - 1 == self.pos_x and sprite.pos_y == self.pos_y) or \
                            (sprite.pos_x + 1 == self.pos_x and sprite.pos_y == self.pos_y):
                        if sprite.error:
                            sprite.fix()
                            return 10
        self.redraw()
        return 0


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


class ExitSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.EXIT_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class MilkSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, speed, sprite_bg):
        super().__init__()
        self.size = size

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.MILK_CAN_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

        self.speed = speed
        self.count = 0

    def redraw(self):
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

    def reset(self):
        self.pos_x = -1
        self.redraw()

    def update(self):
        num_of_rows = int(GlobalConstants.SCREEN_SIZE / GlobalConstants.TILE_SIZE)
        self.count += 1
        if self.count % self.speed == 0:
            self.pos_x += 1
            if self.pos_x > num_of_rows:
                self.pos_x = 0
            self.redraw()
            self.count = 0


class StatusMilkSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.MILK_CAN_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class StatusErrorSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.ERROR_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class TableSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.TABLE_TILE

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size

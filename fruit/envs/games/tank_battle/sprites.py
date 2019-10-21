from fruit.envs.games.tank_battle.constants import GlobalConstants
import numpy as np
import pygame


global_steps = 0


class TankSprite(pygame.sprite.Sprite):

    def __init__(self, size, pos_x, pos_y, sprite_bg, is_enemy, bullet_loading_time, speed, auto_control):
        super().__init__()
        self.size = size  # size
        self.pos_x = pos_x  # current position x
        self.pos_y = pos_y  # current position y
        self.is_enemy = is_enemy  # enemy or ally
        self.loading_time = bullet_loading_time  # loading time of firing a bullet
        self.direction = np.random.randint(0, 4)  # current direction
        self.speed = speed  # speed in pixel
        self.auto_control = auto_control  # human or machine control
        self.fire_started_time = 0  # time of firing
        self.type = GlobalConstants.HARD_OBJECT  # not a bullet
        self.sprite_bg = sprite_bg
        self.image = sprite_bg[self.direction]
        self.rect = self.image.get_rect()
        self.rect.x = self.size * self.pos_x
        self.rect.y = self.size * self.pos_y
        self.target_x = self.pos_x
        self.target_y = self.pos_y
        self.is_in_map = False
        self.is_terminate = False

        if not is_enemy:
            self.direction = GlobalConstants.UP_ACTION

    def update(self):
        self.image = self.sprite_bg[self.direction]
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

        # if self.direction != action:
        #    self.direction = action
        #    return True
        global global_steps
        if not self.is_enemy:
            global_steps = global_steps + 1

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
            if obj.type != GlobalConstants.BULLET_OBJECT and obj.type != GlobalConstants.EXPLOSION_OBJECT:
                if isinstance(obj, TankSprite):
                    if current_x == obj.target_x and current_y == obj.target_y:
                        can_move = False
                        break
                    if current_x == obj.pos_x and current_y == obj.pos_y:
                        can_move = False
                        break
                else:
                    if current_x == obj.pos_x and current_y == obj.pos_y:
                        can_move = False
                        break

        if can_move:
            self.target_x = current_x
            self.target_y = current_y

        return can_move


class BaseSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.HARD_OBJECT

        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x * self.size
        self.rect.y = self.pos_y * self.size


class BulletSprite(pygame.sprite.Sprite):
    def __init__(self, size, tile_size, direction, speed, pos_x, pos_y, owner, sprite_bg):
        super().__init__()

        self.size = size
        self.tile_size = tile_size
        self.direction = direction
        self.speed = speed
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.BULLET_OBJECT
        self.owner = owner
        self.image = sprite_bg

        self.rect = self.image.get_rect()
        adj_pos_x = 0
        adj_pos_y = 0
        if direction == GlobalConstants.LEFT_ACTION:
            adj_pos_x = -self.tile_size / 2
        elif direction == GlobalConstants.RIGHT_ACTION:
            adj_pos_x = self.tile_size / 2
        elif direction == GlobalConstants.UP_ACTION:
            adj_pos_y = -self.tile_size / 2
        elif direction == GlobalConstants.DOWN_ACTION:
            adj_pos_y = self.tile_size / 2
        self.rect.x = pos_x * self.tile_size + self.tile_size / 2 - int(self.size / 2) + adj_pos_x
        self.rect.y = pos_y * self.tile_size + self.tile_size / 2 - int(self.size / 2) + adj_pos_y

    def update(self):
        if self.direction == GlobalConstants.LEFT_ACTION:
            self.rect.x = self.rect.x - self.speed
        elif self.direction == GlobalConstants.RIGHT_ACTION:
            self.rect.x = self.rect.x + self.speed
        elif self.direction == GlobalConstants.UP_ACTION:
            self.rect.y = self.rect.y - self.speed
        elif self.direction == GlobalConstants.DOWN_ACTION:
            self.rect.y = self.rect.y + self.speed


class WallSprite(pygame.sprite.Sprite):
    def __init__(self, size, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.size = size
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.type = GlobalConstants.HARD_OBJECT
        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = pos_x * self.size
        self.rect.y = pos_y * self.size


class ExplosionSprite(pygame.sprite.Sprite):
    def __init__(self, size, abs_x, abs_y, speed, sprites_bg):
        super().__init__()
        self.size = size
        self.images = sprites_bg
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.rect.x = abs_x
        self.rect.y = abs_y
        self.count = 0
        self.current_frame = 0
        self.speed = speed
        self.type = GlobalConstants.EXPLOSION_OBJECT

    def update(self):
        self.count = self.count + 1
        if self.count % self.speed == 0:
            self.image = self.images[self.current_frame]
            self.current_frame = self.current_frame + 1

    def done(self):
        if self.current_frame >= 3:
            return True
        else:
            return False

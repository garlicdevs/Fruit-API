import pygame


class SubmarineSprite(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y, sprite_bg):
        super().__init__()

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x
        self.rect.y = self.pos_y


class TreasureSprite(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y, sprite_bg):
        super().__init__()

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x
        self.rect.y = self.pos_y
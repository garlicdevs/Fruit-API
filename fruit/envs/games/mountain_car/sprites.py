import pygame


class CarSprite(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y, sprite_bg):
        super().__init__()

        self.pos_x = pos_x
        self.pos_y = pos_y

        self.image = sprite_bg
        self.backup_image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x
        self.rect.y = self.pos_y + 25
        self.alpha = 0

    def rotate(self, alpha):
        self.image = pygame.transform.rotate(self.backup_image, alpha)
        self.alpha = alpha

    def restore(self):
        self.image = self.backup_image
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x
        self.rect.y = self.pos_y
        self.alpha = 0


class GoalSprite(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y, sprite_bg):
        super().__init__()
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.image = sprite_bg
        self.rect = self.image.get_rect()
        self.rect.x = self.pos_x
        self.rect.y = self.pos_y
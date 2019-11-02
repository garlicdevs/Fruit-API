import pygame


class ResourceManager(object):
    LAND_TILE = "land4.png"
    PLANT_TILE = "plant.png"
    KEY_TILE = "key.png"
    MINUS_TILE = 'minus.png'
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

    def get_resources(self):
        return [ResourceManager.LAND_TILE, ResourceManager.PLANT_TILE, ResourceManager.KEY_TILE,
                ResourceManager.GUY_L_1_TILE, ResourceManager.GUY_L_2_TILE, ResourceManager.GUY_R_1_TILE,
                ResourceManager.GUY_R_2_TILE, ResourceManager.GUY_U_1_TILE, ResourceManager.GUY_U_2_TILE,
                ResourceManager.GUY_D_1_TILE, ResourceManager.GUY_D_2_TILE, ResourceManager.MINUS_TILE]

    def __add_resources(self):
        resources = self.get_resources()
        for resource in resources:
            image = pygame.image.load(self.current_path + resource)
            if self.render:
                image = pygame.transform.scale(image, (self.tile_size, self.tile_size)).convert_alpha()
            else:
                image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
            self.resources[resource] = image

    def get_image(self, key):
        return self.resources[key]

    def get_font(self):
        return self.font
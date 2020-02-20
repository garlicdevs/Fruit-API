import cocos
import cocos.collision_model as cm
import cocos.euclid as eu
from cocos.rect import Rect

from mazeexp.engine import config


def world_to_view(v):
    """world coords to view coords; v an eu.Vector2, returns (float, float)"""
    return v.x * config.scale_x, v.y * config.scale_y

#def reflection_y(a):
#    assert isinstance(a, eu.Vector2)
#    return eu.Vector2(a.x, -a.y)


class Collidable(cocos.sprite.Sprite):
    palette = {}  # injected later
    """
    Collidable

    Responsabilities:
        Generate a collision manager sprite
    """

    def __init__(self, cx, cy, radius, btype, img, removable=False):
        super(Collidable, self).__init__(img)

        # TODO: Inheritable moving items `velocity=None`

        self.palette = config.settings['view']['palette']

        self.radius = radius
        # the 1.05 so that visual radius a bit greater than collision radius
        # FIXME: Both `scale_x` and `scale_y`
        self.scale = (self.radius * 1.05) * config.scale_x / (self.image.width / 2.0)
        self.btype = btype
        self.color = self.palette[btype]
        self.cshape = cm.CircleShape(eu.Vector2(cx, cy), self.radius)
        self.update_center(self.cshape.center)

        self.removable = removable

    def update_center(self, cshape_center):
        """cshape_center must be eu.Vector2"""
        assert isinstance(cshape_center, eu.Vector2)

        self.position = world_to_view(cshape_center)
        self.cshape.center = cshape_center

    def get_rect(self):
        ppos = self.cshape.center
        r = self.cshape.r

        # FIXME: Use top, bottom, left, right
        return Rect(ppos.x-r, ppos.y-r, r*2, r*2)

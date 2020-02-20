import random

import cocos.collision_model as cm
import cocos.euclid as eu

from mazeexp.engine import config
from mazeexp.engine.collidable import Collidable


class WorldItems(object):
    """
    WorldItems

    Methods inherited by WorldLayer
    Has context for game settings, map state and player state

    Responsabilities:
        Add rewards to player in response to game events
    """

    def __init__(self):
        super(WorldItems, self).__init__()

        player = config.settings['player']
        world = config.settings['world']

        self.pics = config.pics
        self.to_remove = []

        # TODO: Only create if needed by game `mode`
        cell_size = player['radius'] * 0.25
        self.collman = cm.CollisionManagerGrid(0.0, world['width'],
                                               0.0, world['height'],
                                               cell_size, cell_size)

    def create_items(self):
        """
        Create collidable items
        """
        if not self.mode['items'] or len(self.mode['items']) == 0: return

        # FIXME: Check if already contains
        self.collman.add(self.player)

        for k in self.mode['items']:
            item = self.mode['items'][k]
            #{'terminal': False, 'num': 50, 'scale': 1.0, 'reward': 2.0}
            radius = item['scale'] * self.player.radius
            for i in range(item['num']):
                self.add_item(radius, k)

            # add gate
            #rGate = gate_scale * rPlayer
            #self.gate = Player(cx, cy, rGate, 'gate', pics['wall'])
            #self.gate.color = Player.palette['wall']
            #cntTrys = 0
            #while cntTrys < 100:
            #    cx = rGate + random.random() * (width - 2.0 * rGate)
            #    cy = rGate + random.random() * (height - 2.0 * rGate)
            #    self.gate.update_center(eu.Vector2(cx, cy))
            #    if not self.collman.they_collide(self.player, self.gate):
            #        break
            #    cntTrys += 1
            #self.add(self.gate, z=z)
            #z += 1
            #self.collman.add(self.gate)



    def add_item(self, radius, item_type):
        """
        Add a single item in random open position
        """
        assert isinstance(radius, int) or isinstance(radius, float)
        assert isinstance(item_type, str)

        separation_scale = 1.1
        min_separation = separation_scale * radius

        # Removable item
        item = Collidable(0, 0, radius, item_type, self.pics[item_type], True)
        cntTrys = 0
        while cntTrys < 100:
            cx = radius + random.random() * (self.width - 2.0 * radius)
            cy = radius + random.random() * (self.height - 2.0 * radius)

            # Test if colliding with wall.
            # Top left
            cells = []
            cells.append(self.map_layer.get_at_pixel(cx-radius, cy-radius))
            # Top right
            cells.append(self.map_layer.get_at_pixel(cx+radius, cy-radius))
            # Bottom  left
            cells.append(self.map_layer.get_at_pixel(cx-radius, cy+radius))
            # Bottom  right
            cells.append(self.map_layer.get_at_pixel(cx+radius, cy+radius))

            wall = False
            for cell in cells:
                wall = cell and cell.tile and cell.tile.id > 0
                if wall:
                    break

            if wall:
                continue

            item.update_center(eu.Vector2(cx, cy))
            if self.collman.any_near(item, min_separation) is None:
                self.add(item, z=self.z)
                self.z += 1
                self.collman.add(item)
                break
            cntTrys += 1

    def update_collisions(self):
        """
        Test player for collisions with items
        """
        if not self.mode['items'] or len(self.mode['items']) == 0: return

        # update collman
        # FIXME: Why update each frame?
        self.collman.clear()
        for z, node in self.children:
            if hasattr(node, 'cshape') and type(node.cshape) == cm.CircleShape:
                self.collman.add(node)

        # interactions player - others
        for other in self.collman.iter_colliding(self.player):
            typeball = other.btype
            self.logger.debug('collision', typeball)

            # TODO: Limit player position on non-removable items
            #if not other.removable:
            #    pass

            if other.removable:
                self.to_remove.append(other)

            self.reward_item(typeball)

        #
        #    elif (typeball == 'wall' or
        #          typeball == 'gate' and self.cnt_food > 0):
        #        self.level_losed()
        #
        #    elif typeball == 'gate':
        #        self.level_conquered()

        self.remove_items()

    def remove_items(self):
        # at end of frame do removes; as collman is fully regenerated each frame
        # theres no need to update it here.
        while len(self.to_remove) > 0:
            self.remove(self.to_remove.pop())

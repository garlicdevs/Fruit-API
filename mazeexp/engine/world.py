import logging

from mazeexp.engine import config
from mazeexp.engine.generator import Generator
from mazeexp.engine.player import Player
from mazeexp.engine.score import ScoreLayer
from mazeexp.engine.world_items import WorldItems
from mazeexp.engine.world_queries import WorldQueries
from mazeexp.engine.world_rewards import WorldRewards

logging.basicConfig()

import math
from random import randint

import cocos
import cocos.euclid as eu
import cocos.mapcolliders as mc
import cocos.tiles as ti
from cocos import draw


import os
script_dir = os.path.dirname(__file__)

class WorldLayer(WorldItems, WorldQueries, WorldRewards, cocos.layer.Layer, mc.RectMapCollider):

    """
    WorldLayer

    Responsabilities:
        Generation: random generates a level
        Initial State: Set initial playststate
        Play: updates level state, by time and user input. Detection of
        end-of-level conditions.
        Level progression.
    """
    is_event_handler = True

    def __init__(self, mode_id = 0, fn_show_message=None):
        super(WorldLayer, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.settings['log_level'])

        self.mode_id = mode_id
        self.mode = config.modes[self.mode_id]
        self.force_fps = config.settings['world']['force_fps']

        # Time it takes to travel half a square at full speed
        self.consumed_dt = config.settings['player']['top_speed'] / min(config.tiles['th'], config.tiles['tw']) / 2

        self.fn_show_message = fn_show_message

        self.z = 0

        # basic geometry
        world = config.settings['world']
        self.width = world['width']  # world virtual width
        self.height = world['height']  # world virtual height

        self.tiles_w = config.tiles['width']
        self.tiles_h = config.tiles['height']

        self.generator = Generator()

        self.bindings = world['bindings']
        buttons = {}
        for k in self.bindings:
            buttons[self.bindings[k]] = 0
        self.buttons = buttons

        self.on_bump_handler = self.on_bump_slide

        self.schedule(self.update)
        self.ladder_begin()

    def ladder_begin(self):
        self.level_num = 0
        self.empty_level()
        #msg = 'Maze Explorer'
        #self.fn_show_message(msg, callback=self.level_launch)
        self.level_launch()

    def level_launch(self):
        self.generate_random_level()
        #msg = 'level %d' % self.level_num
        #self.fn_show_message(msg, callback=self.level_start)
        self.level_start()

    def level_start(self):
        self.win_status = 'undecided'

    def level_conquered(self):
        self.win_status = 'intermission'
        msg = 'level %d\nconquered !' % self.level_num
        # TODO: Set `done`.
        self.fn_show_message(msg, callback=self.level_next)

    def level_losed(self):
        self.win_status = 'losed'
        msg = 'ouchhh !!!'
        # TODO: Set `done`.
        self.fn_show_message(msg, callback=self.ladder_begin)

    def level_next(self):
        self.empty_level()
        self.level_num += 1
        self.level_launch()

    def empty_level(self):
        # del old actors, if any
        for node in self.get_children():
            self.remove(node)
        assert len(self.children) == 0
        self.player = None
        self.gate = None
        self.to_remove = []

        self.win_status = 'intermission'  # | 'undecided' | 'conquered' | 'losed'

        # player phys params
        #if self.player is Player:
        #    self.player.reset()

    def generate_random_level(self):
        """
        Configure and add cocos layers
        """

        # build !
        width = self.width
        height = self.height
        tiles_w = self.tiles_w
        tiles_h = self.tiles_h
        self.z = 0

        # add walls
        #self.map_layer = ti.load(os.path.join(script_dir, 'test.tmx'))['map0']
        self.map_layer = self.generator.map(tiles_w, tiles_h)
        self.map_layer.set_view(0, 0, self.map_layer.px_width, self.map_layer.px_height)
        # FIXME: Both `scale_x` and `scale_y`
        self.map_layer.scale = config.scale_x
        self.add(self.map_layer, z=self.z)
        self.z += 1

        # add floor
        # TODO: Move to `Generator.inverse(map)`
        self.visit_layer = ti.load(os.path.join(script_dir, 'assets', 'ones.tmx'))['map0']
        for i in range(0, len(self.map_layer.cells)):
            for j in range(0, len(self.map_layer.cells[i])):
                col = self.map_layer.cells[i][j]
                # If wall exists, remove floor
                if col.tile and col.tile.id > 0:
                    self.visit_layer.cells[i][j].tile = None

        self.visit_layer.set_view(0, 0, self.visit_layer.px_width, self.visit_layer.px_height)
        # FIXME: Both `scale_x` and `scale_y`
        self.visit_layer.scale = config.scale_x
        self.add(self.visit_layer, z=-1)

        # add player
        # Start in random corner
        corner = randint(0,3)
        padding = eu.Vector2(self.map_layer.tw*1.5, self.map_layer.th*1.5)
        corners = [
            eu.Vector2(padding.x, padding.y), # Bottom left
            eu.Vector2(((tiles_w+1)*self.map_layer.tw)-padding.x, padding.y), # Bottom right
            eu.Vector2(padding.x, ((tiles_h+1)*self.map_layer.th)-padding.y), # Top right
            eu.Vector2(((tiles_w+1)*self.map_layer.tw)-padding.x, ((tiles_h+1)*self.map_layer.th)-padding.y) # Top left
        ]
        rotations = [
            45,
            -45,
            135,
            -135
        ]
        self.spawn = corners[corner]
        self.player = Player(self.spawn.x, self.spawn.y)
        self.player.rotation = rotations[corner]
        self.add(self.player, z=self.z)
        self.z += 1

        self.score = ScoreLayer(self.player.stats)
        self.add(self.score, z=self.z)
        self.z += 1

        # Draw sensors
        # TODO: Decouple into view rendering
        a = math.radians(self.player.rotation)
        for sensor in self.player.sensors:
            rad = a + sensor.angle
            start = self.player.cshape.center
            end = start.copy()
            end.x += math.sin(rad) * sensor.proximity
            end.y += math.cos(rad) * sensor.proximity
            sensor.line = draw.Line(start, end, self.player.palette['wall'] + (int(255*0.5),))
            self.map_layer.add(sensor.line)

        # Generate obstacles
        self.create_items()

    def update(self, dt):
        """
        Updates game engine each tick
        """
        # if not playing dont update model
        if self.win_status != 'undecided':
            return

        # Step known time for agents
        if self.force_fps > 0:
            dt = 1 / self.force_fps

        # update target
        self.player.update_rotation(dt, self.buttons)

        # Get planned update
        newVel = self.player.do_move(dt, self.buttons)

        # Position collision rects
        oldRect = self.player.get_rect()
        newRect = oldRect.copy()

        oldPos = self.player.cshape.center
        remaining_dt = dt
        newPos = oldPos.copy()

        # So WorldLayer is given `bumped` attributes at startup.
        if dt == 0:
            newVel.x, newVel.y = self.collide_map(self.map_layer, oldRect, oldRect, newVel.x, newVel.y)

        # Get planned velocity result
        def get_new_rect(pos, width, height, dt, velocity):
            targetPos = pos + dt * velocity
            x = targetPos.x - width/2
            y = targetPos.y - height/2
            return x, y

        while remaining_dt > 1.e-6:
            #print('remaining_dt', remaining_dt)
            newRect.x, newRect.y = get_new_rect(oldPos, newRect.width, newRect.height, remaining_dt, newVel)
            newVel.x, newVel.y = self.collide_map(self.map_layer, oldRect, newRect, newVel.x, newVel.y)

            remaining_dt -= self.consumed_dt

        newRect.x, newRect.y = get_new_rect(oldPos, newRect.width, newRect.height, dt, newVel)

        # Ensure player can't escape borders
        border = False
        if newRect.top > self.height:
            newRect.top = self.height
            border = True
        if newRect.bottom < self.map_layer.th:
            newRect.bottom = self.map_layer.th
            border = True
        if newRect.left < self.map_layer.th:
            newRect.left = self.map_layer.th
            border = True
        if newRect.right > self.width:
            newRect.right = self.width
            border = True

        newPos = self.player.cshape.center
        newPos.x, newPos.y = newRect.center
        self.player.velocity = newVel
        self.player.update_center(newPos)

        # Collision detected
        if border or self.bumped_x or self.bumped_y:
            #print('bumped')
            self.reward_wall()

        # In WorldLayer so we can access map
        self.update_visited()
        self.update_sensors()

        self.reward_battery()
        self.reward_proximity()

        # TODO: Display messages for humans at some point
        #if self.player.game_over:
        #    self.level_losed()

        self.update_collisions()

        # update collman
        #self.collman.clear()
        #for z, node in self.children:
        #    self.collman.add(node)

        # interactions player - others
        #for other in self.collman.iter_colliding(self.player):
        #    print('collman', other)
        #    typeball = other.btype

        #    if typeball == 'food':
        #        self.toRemove.add(other)
        #        self.cnt_food -= 1
        #        if not self.cnt_food:
        #            self.open_gate()
        #
        #    elif (typeball == 'wall' or
        #          typeball == 'gate' and self.cnt_food > 0):
        #        self.level_losed()
        #
        #    elif typeball == 'gate':
        #        self.level_conquered()



        # at end of frame do removes; as collman is fully regenerated each frame
        # theres no need to update it here.
    #    for node in self.toRemove:
    #        self.remove(node)
    #    self.toRemove.clear()

    def update_visited(self):
        """
        Updates exploration map visited status
        """
        assert isinstance(self.player.cshape.center, eu.Vector2)
        pos = self.player.cshape.center

        # Helper function
        def set_visited(layer, cell):
            if cell and not cell.properties.get('visited') and cell.tile and cell.tile.id > 0:
                cell.properties['visited'] = True

                self.reward_explore()

                # TODO: Decouple into view rendering
                # Change colour of visited cells
                key = layer.get_key_at_pixel(cell.x, cell.y)
                #layer.set_cell_color(key[0], key[1], [155,155,155])
                layer.set_cell_opacity(key[0], key[1], 255*0.8)
        # End Helper

        # Get the current tile under player
        current = self.visit_layer.get_at_pixel(pos.x, pos.y)

        if current:
            # In spawn square
            if current == self.visit_layer.get_at_pixel(self.spawn.x, self.spawn.y):
                self.reward_goal()

            # Only record/reward exploration when battery is above 50%
            #if self.player.stats['battery'] > 50:
            set_visited(self.visit_layer, current)
            neighbours = self.visit_layer.get_neighbors(current)
            for cell in neighbours:
                neighbour = neighbours[cell]
                set_visited(self.visit_layer, neighbour)

    def update_sensors(self):
        """
        Check path for each sensor and record wall proximity
        """
        assert isinstance(self.player.cshape.center, eu.Vector2)
        pos = self.player.cshape.center

        a = math.radians(self.player.rotation)
        for sensor in self.player.sensors:
            sensor.sensed_type = 'wall'
            rad = a + sensor.angle
            dis = min(self.distance_to_tile(pos, rad), sensor.max_range)

            # Keep state of sensed range, `dis` is from center
            sensor.proximity = dis - self.player.radius

            # Check for collisions with items
            # List of items within sensor range, do for each sensor's range
            if self.mode['items'] and len(self.mode['items']) > 0:
                nears = self.collman.ranked_objs_near(self.player, sensor.max_range)
                for near in nears:
                    other, other_dis = near
                    # Distances are from edge to edge see #2
                    other_dis += self.player.radius
                    # Skip if further
                    if other_dis > dis:
                        continue

                    # Determine if within `fov`
                    other_rad = math.atan2(other.x - self.player.x, other.y - self.player.y)
                    # Round to bearing within one revolution
                    other_rad = other_rad % (math.pi*2)
                    round_rad = rad % (math.pi*2)
                    if abs(other_rad - round_rad) < (sensor.fov/2):
                        sensor.proximity = other_dis - self.player.radius
                        sensor.sensed_type = other.btype
                        dis = other_dis

            # Redirect sensor lines
            # TODO: Decouple into view rendering
            end = pos.copy()
            end.x += math.sin(rad) * dis
            end.y += math.cos(rad) * dis
            sensor.line.start = pos
            sensor.line.end = end
            sensor.line.color = self.player.palette[sensor.sensed_type] + (int(255*0.5),)

    def get_state(self):
        """
        Create state from sensors and battery
        """
        # Include battery level in state
        battery = self.player.stats['battery']/100
        # Create observation from sensor proximities
        # TODO: Have state persist, then update columns by `sensed_type`

        # Multi-channel; detecting `items`
        if len(self.mode['items']) > 0:
            observation = []
            for sensor in self.player.sensors:
                col = []
                # Always include range in channel 0
                col.append(sensor.proximity_norm())
                for item_type in self.mode['items']:
                    if sensor.sensed_type == item_type:
                        col.append(sensor.proximity_norm())
                    else:
                        # Default to 1 (`max_range/max_range`)
                        col.append(1)
                observation.append(col)
            if 'battery' in self.mode:
                observation.append([battery,1,1])

        # Single-channel; walls only
        else:
            observation = [o.proximity_norm() for o in self.player.sensors]
            if 'battery' in self.mode:
                observation.append(battery)

        return observation

    #def open_gate(self):
    #    self.gate.color = Player.palette['gate']

    def on_key_press(self, k, m):
        binds = self.bindings
        if k in binds:
            self.buttons[binds[k]] = 1
            return True
        return False

    def on_key_release(self, k, m):
        binds = self.bindings
        if k in binds:
            self.buttons[binds[k]] = 0
            return True
        return False

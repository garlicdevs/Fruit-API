from __future__ import division, print_function, unicode_literals

import pyglet
import cocos
from cocos.director import director

from mazeexp.engine import config
from mazeexp.engine.message import MessageLayer
from mazeexp.engine.world import WorldLayer


class MazeExplorer():
    """
    MazeExplorer

    Wrapper for game engine
    """

    def __init__(self, mode_id=0, visible = True):
        config.settings['window']['visible'] = visible

        self.mode_id = int(mode_id)
        self.mode = config.modes[self.mode_id]

        self.director = director
        self.director.init(**config.settings['window'])
        #pyglet.font.add_directory('.') # adjust as necessary if font included
        self.z = 0

        self.actions_num = len(config.settings['player']['actions'])
        # Sensors
        self.observation_num = config.settings['player']['sensors']['num']
        # Plus one for battery indicator
        if 'battery' in self.mode:
            self.observation_num += 1
        # Observation channels as game mode requires, plus one for walls
        self.observation_chans = len(self.mode['items']) + 1

    def reset(self):
        """
        Attach a new engine to director
        """
        self.scene = cocos.scene.Scene()
        self.z = 0

        palette = config.settings['view']['palette']
        #Player.palette = palette
        r, g, b = palette['bg']
        self.scene.add(cocos.layer.ColorLayer(r, g, b, 255), z=self.z)
        self.z += 1
        message_layer = MessageLayer()
        self.scene.add(message_layer, z=self.z)
        self.z += 1
        self.world_layer = WorldLayer(self.mode_id, fn_show_message=message_layer.show_message)
        self.scene.add(self.world_layer, z=self.z)
        self.z += 1

        self.director._set_scene(self.scene)

        # Step once to refresh before `act`
        self.step()

        # TODO: Reset to `ones`?
        return self.world_layer.get_state()

    def act(self, action):
        """
        Take one action for one step
        """
        # FIXME: Hack to change in return type
        action = int(action)
        assert isinstance(action, int)
        assert action < self.actions_num, "%r (%s) invalid"%(action, type(action))

        # Reset buttons
        for k in self.world_layer.buttons:
            self.world_layer.buttons[k] = 0

        # Apply each button defined in action config
        for key in self.world_layer.player.controls[action]:
            if key in self.world_layer.buttons:
                self.world_layer.buttons[key] = 1

        # Act in the environment
        self.step()

        observation = self.world_layer.get_state()
        reward = self.world_layer.player.get_reward()
        terminal = self.world_layer.player.game_over
        info = {}

        return observation, reward, terminal, info

    def step(self):
        """
        Step the engine one tick
        """
        self.director.window.switch_to()
        self.director.window.dispatch_events()
        self.director.window.dispatch_event('on_draw')
        self.director.window.flip()

        # Ticking before events caused glitches.
        pyglet.clock.tick()

        #for window in pyglet.app.windows:
        #    window.switch_to()
        #    window.dispatch_events()
        #    window.dispatch_event('on_draw')
        #    window.flip()

    def run(self):
        """
        Run in real-time
        """
        self.reset()
        return self.director.run(self.scene)

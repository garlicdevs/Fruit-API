import math
import logging

import pyglet
from pyglet.window import key

import os
script_dir = os.path.dirname(__file__)

tiles = {
    "tw": 10,
    "th": 10,
    # Current template arrangement will take up to 50x50 tiles
    "width": 20,
    "height": 20
}

#fe = 1.0e-4
settings = {
    "log_level": logging.INFO,
    "window": {
        "width": 500,
        "height": 500,
        "vsync": True,
        "resizable": False
    },
    "player": {
        "radius": tiles['tw'] / 4,
        # TODO: Put all velocity settings in dict
        "top_speed": 10.0,
        "angular_velocity": 60.0,  # degrees / s
        "accel": 5.0,
        "deaccel": 15.0,
        # TODO: Refactor to action `costs`, will apply to whichever `stat`
        "battery_use": {
            "angular": 0.01,
            "linear": 0.01
        },
        "sensors": {
            "num": 9,
            "fov": 15*math.pi/180,
            "max_range": 200 / 4
        },
        "actions": [
            #['noop'],
            ['left'],
            ['up', 'left'],
            ['up'],
            ['up', 'right'],
            ['right']
        ]
    },
    "world": {
        "force_fps": 5.0, # Used by agents to step velocity updates
        "width": tiles['tw'] * tiles['width'],
        "height": tiles['th'] * tiles['height'],
        "bindings": {
            #0: 'noop',
            key.LEFT: 'left',
            key.RIGHT: 'right',
            key.UP: 'up',
        }
    },
    "view": {
        # as the font file is not provided it will decay to the default font;
        # the setting is retained anyway to not downgrade the code
        "font_name": 'Axaxax',
        "palette": {
            'bg': (0, 65, 133),
            'wall': (50, 50, 100), # Wall sensor colour
            'player': (237, 27, 36),
            #'wall': (247, 148, 29),
            'gate': (140, 198, 62),
            'food': (140, 198, 62),
            'poison': (198, 62, 62)
        }
    }
}

# Callbacks to test reward conditions
def __cond_action_up(world):
    # Detect other keys
    held_buttons = {k:v for k,v in world.buttons.iteritems() if v == 1}
    return len(held_buttons) == 1 and world.buttons['up'] == 1

def __cond_battery_out(world):
    return world.player.stats['battery'] <= 0

def __cond_explore_battery(world):
    return world.player.stats['battery'] > 50

def __cond_goal_battery(world):
    return world.player.stats['battery'] <= 50

modes = [
    # Mode 0
    {
        #"proximity": {
        #    "cond": __cond_action_up,
        #    "reward": 1.1 # 10% Bonus on proximity reward
        #},
        "wall": {
            "reward": -10.0,
            "terminal": False
        },
        "items": {
            "food": {
                "num": 20,
                "scale": 2.0,
                "reward": 5.0,
                "terminal": False
            },
            "poison": {
                "num": 20,
                "scale": 2.0,
                "reward": -6.0,
                "terminal": False
            },
        }
    },
    # Mode 1
    {
        "battery": {
            "cond": __cond_battery_out,
            "reward": -100.0,
            "terminal": True
        },
        "explore": {
            "cond": __cond_explore_battery,
            "reward": 1.0,
            "terminal": False
        },
        "goal": {
            "cond": __cond_goal_battery,
            "reward": 200.0,
            "terminal": True
        },
        "wall": {
            "reward": -100.0,
            "terminal": True
        },
        "items": {}
    }
]

# Ensure all modes have items for convenience
for mode in modes:
    if mode['items'] is None:
        mode['items'] = {}

# world to view scales
scale_x = settings["window"]["width"] / settings["world"]["width"]
scale_y = settings["window"]["height"] / settings["world"]["height"]

# load resources:
pics = {
    "player": pyglet.image.load(os.path.join(script_dir, 'assets', 'player7.png')),
    "food": pyglet.image.load(os.path.join(script_dir, 'assets', 'circle6.png')),
    "poison": pyglet.image.load(os.path.join(script_dir, 'assets', 'circle6.png'))
}

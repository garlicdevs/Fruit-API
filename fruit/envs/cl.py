import random
import time
import os
from collections import deque
import pygame
import numpy as np
from fruit.envs.base import BaseEnvironment
import carla


# Please note that this environment is very basic, users should add more features for it
# Please see manual_control.py in Carla example codes
class CarlaEnvironment(BaseEnvironment):
    def __init__(self, width=400, height=300, render=False, state_processor=None, autopilot=True,
                 buffer_size=5, num_of_cars=1, npc_cars=10, fps=60, world='Town01', camera='rgb'):
        self.should_render = render
        self.actor_list = []
        self.width = width
        self.height = height
        self.processor = state_processor
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.time_steps = 0
        self.num_of_agents = num_of_cars
        self.npc_cars = npc_cars
        self.num_frames = 0
        self.fps = fps
        self.current_state = None
        self.npc_cars = npc_cars
        self.world = world
        self.camera_name = camera
        self.fines = deque(maxlen=100)
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.load_world(world)
        self.terminate_state = False
        self.autopilot = autopilot

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1/self.fps
        self.world.apply_settings(settings)

        self.initialize()

        if self.should_render:
            self.__init_pygame_engine()

    def get_game_name(self):
        return 'CARLA ENVIRONMENT'

    def initialize(self):
        blueprint_library = self.world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(bp, transform)
        print('Create main vehicle %s' % self.vehicle.type_id)
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(self.autopilot)

        camera_bp = blueprint_library.find('sensor.camera.' + self.camera_name)
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        camera_bp.set_attribute('fov', '110')
        camera_bp.set_attribute('sensor_tick', '0.')
        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)

        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, z=2))
        self.collision = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision)

        invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        invasion_transform = carla.Transform(carla.Location(x=0, z=2))
        self.invasion = self.world.spawn_actor(invasion_bp, invasion_transform, attach_to=self.vehicle)
        self.actor_list.append(self.invasion)

        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, self.npc_cars):
            transform.location.x += 8.0
            bp = random.choice(blueprint_library.filter('vehicle'))
            npc = self.world.try_spawn_actor(bp, transform)
            if npc is not None:
                self.actor_list.append(npc)
                npc.set_autopilot()
                print('created %s' % npc.type_id)

        self.camera.listen(self.__add_to_buffer)
        self.collision.listen(self.__collision)
        self.invasion.listen(self.__invasion)

    def __collision(self, event):
        print('Collision {} !'.format(event))
        self.terminal_state = True

    def __invasion(self, event):
        print('Lane Invasion {} !'.format(event))
        self.fines.append(-100)

    def __init_pygame_engine(self):
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pygame.init()

        pygame.display.set_caption(self.get_game_name())
        self.screen = pygame.display.set_mode((self.width, self.height))

    def __add_to_buffer(self, image):
        print('Frame is coming ..')
        self.num_frames += 1
        self.buffer.append(image)

    def clone(self):
        return CarlaEnvironment(width=self.width, height=self.height, render=self.should_render,
                                state_processor=self.processor, buffer_size=self.buffer_size, autopilot=self.autopilot,
                                num_of_cars=self.num_of_agents, npc_cars=self.npc_cars, fps=self.fps,
                                world=self.world, camera=self.camera)

    def step(self, actions):
        if not self.autopilot:
            self.vehicle.apply_control(carla.VehicleControl(throttle=actions[0], steer=actions[1], brake=actions[2],
                                                            hand_brake=actions[3], reverse=actions[4],
                                                            manual_gear_shift=actions[5], gear=actions[6]))
        else:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1, steer=0, brake=0,
                                     hand_brake=0, reverse=0,
                                     manual_gear_shift=0, gear=0))
        if len(self.fines) <= 0:
            return 1
        else:
            return self.fines.pop()

    def reset(self):
        self.time_steps = 0
        self.num_frames = 0
        self.terminal_state = False
        self.fines.clear()
        self.buffer.clear()

        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

        self.initialize()

        # Wait for the first frame
        while True:
            time.sleep(0.01)
            if len(self.buffer) <= 0:
                continue
            else:
                break

        return self.get_state()

    def get_current_steps(self):
        return self.time_steps

    def get_action_space(self):
        from fruit.types.priv import Space
        return tuple([Space(0.0, 1.0, False),
                      Space(-1.0, 1.0, False),
                      Space(0.0, 1.0, False),
                      Space(0, 1, True),
                      Space(0, 1, True),
                      Space(0, 1, True),
                      Space(0, 2, True)])

    def get_state_space(self):
        from fruit.types.priv import Space
        shape = (self.width, self.height, 3)
        min_value = np.zeros(shape)
        max_value = np.full(shape, 255)
        return Space(min_value, max_value, True)

    def step_all(self, action):
        reward = self.step(action)
        self.get_state()
        terminal = self.is_terminal()
        return self.current_state, reward, terminal, None

    def get_state(self):
        if len(self.buffer) > 0:
            observation = self.buffer.pop().raw_data
            array = np.frombuffer(observation, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.height, self.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            observation = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            if self.processor is not None:
                self.current_state = self.processor.process(observation)
            else:
                self.current_state = observation
        return self.current_state

    def is_terminal(self):
        return self.terminal_state

    def is_atari(self):
        return False

    def is_render(self):
        return self.should_render

    def get_number_of_objectives(self):
        return self.num_of_agents

    def get_number_of_agents(self):
        return self.num_of_agents

    def get_processor(self):
        return self.processor

    def render(self):
        print('Render ..')
        if self.should_render and self.current_state is not None:

            self.screen.blit(self.current_state, (0, 0))

            pygame.display.flip()
        time.sleep(1/self.fps)


def get_random_action(is_discrete, action_range, action_space):
    if is_discrete:
        if len(action_range) == 2 and isinstance(action_range[0], (list, np.ndarray, tuple)):
            action = [random.randint(action_range[0][i], action_range[1][i]) for i in range(len(action_range[0]))]
        else:
            action = random.randint(0, len(action_range) - 1)
    else:
        rand = np.random.rand(*tuple(action_space.get_shape()))[0]
        action = np.multiply(action_range[1] - action_range[0], rand) + action_range[0]
    return action


def create_random_agent():
    fruit_env = CarlaEnvironment(render=True)

    state_space = fruit_env.get_state_space()
    if isinstance(state_space, tuple):
        for s in state_space:
            print(s.get_range(), s.get_shape())
    else:
        print(state_space.get_range(), state_space.get_shape())

    action_space = fruit_env.get_action_space()
    is_discrete = False
    action_range = None
    if isinstance(action_space, tuple):
        for s in action_space:
            action_range, is_discrete = s.get_range()
            print(action_range, s.get_shape())
    else:
        action_range, is_discrete = action_space.get_range()
        print(action_range, action_space.get_shape())

    fruit_env.reset()
    for i in range(1000):
        if isinstance(action_space, tuple):
            action = []
            for s in action_space:
                action_range, is_discrete = s.get_range()
                action.append(get_random_action(is_discrete, action_range, s))
        else:
            action = get_random_action(is_discrete, action_range, action_space)

        reward = fruit_env.step(action)
        fruit_env.render()
        next_state = fruit_env.get_state()
        state = next_state
        terminal = fruit_env.is_terminal()
        print(action, reward)

        if terminal:
            fruit_env.reset()
            break


if __name__ == '__main__':
    create_random_agent()

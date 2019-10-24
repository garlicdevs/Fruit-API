import collections as cl
from fruit.state.processor import Processor
from fruit.utils.image import *


class SeaquestMapProcessor(Processor):
    def __init__(self):
        self.resize_shape = (84, 84)
        self.convert_to_grayscale = True
        self.oxy_base_line = 172
        self.oxy_detect_point = 71
        self.human_lives_base_line = 181
        self.human_pos = [63, 71, 79, 87, 95, 103]
        self.current_lives = 0
        self.human_rewards = 0
        self.is_low_oxygen = False
        self.refill_rewards = 0
        self.h_rewards = cl.deque(maxlen=100)
        self.r_rewards = cl.deque(maxlen=100)
        self.l_rewards = cl.deque(maxlen=100)
        self.is_first = True
        self.agent_lives_base_line = 27
        self.agent_lives_detect_x = 78
        self.goals = cl.deque(maxlen=100)
        self.map_data = convert_rgb_to_grayscale(np.empty((210, 160, 3), dtype=np.uint8))
        self.count = 0

    def clone(self):
        return SeaquestMapProcessor()

    def set_resize_shape(self, resize_shape):
        self.resize_shape = resize_shape

    def set_grayscale(self, convert_to_grayscale):
        self.convert_to_grayscale = convert_to_grayscale

    def get_grayscale(self):
        return self.convert_to_grayscale

    def get_resize_shape(self):
        return self.resize_shape

    def reset(self):
        self.is_low_oxygen = False
        self.human_rewards = 0
        self.refill_rewards = 0
        self.is_first = True
        self.current_lives = 0
        self.map_data = convert_rgb_to_grayscale(np.empty((210, 160, 3), dtype=np.uint8))

    def check_data(self, img):

        self.goals.clear()

        prev = self.current_lives
        self.current_lives = 0
        for x in self.human_pos:
            if img[self.human_lives_base_line][x] < 100:
                self.current_lives = self.current_lives + 1

        if self.current_lives == prev + 1:
            self.human_rewards = self.human_rewards + 1
            self.h_rewards.append(1)

        if self.current_lives == prev - 1:
            if img[self.agent_lives_base_line][self.agent_lives_detect_x] > 100:
                if prev == 6:
                    self.refill_rewards = self.refill_rewards + 1
                    self.r_rewards.append(1)
                else:
                    self.refill_rewards = self.refill_rewards + 1
                    self.r_rewards.append(1)

        if self.current_lives < 6:
            if img[self.oxy_base_line][self.oxy_detect_point] < 100:
                self.goals.append([49, 85])
            else:
                for y in range(60, 150, 4):
                    for x in range(10, 150, 4):
                        if img[y][x] == 84:
                            self.goals.append([y, x])
        else:
            self.goals.append([49, 85])

        if img[self.oxy_base_line][self.oxy_detect_point] < 100:
            self.l_rewards.append(1)

        # save_grayscale_image(img, full_path="/home/garlicdevs/Desktop/Images/image_" + str(self.count) + ".png")
        # map = self.get_map()
        # save_grayscale_image(map, full_path="/Users/alpha/Desktop/images/image" + str(self.count) + ".png")
        self.count = self.count + 1

    def process(self, pre_image):

        gray_scale = convert_rgb_to_grayscale(pre_image)

        self.check_data(gray_scale)

        resize_img = resize_grayscale_image(gray_scale, self.resize_shape)

        return resize_img

    def get_rewards(self):
        if len(self.h_rewards) > 0:
            r1 = self.h_rewards.pop()
        else:
            r1 = 0
        if len(self.r_rewards) > 0:
            r2 = self.r_rewards.pop()
        else:
            r2 = 0
        if len(self.l_rewards) > 0:
            r3 = self.l_rewards.pop()
        else:
            r3 = 0
        return [r1, r2, r3, self.current_lives]

    def get_num_of_objectives(self):
        return 3

    def get_map(self):
        self.map_data.fill(0)

        for g in self.goals:
            self.map_data[g[0]][g[1]] = 255
            self.map_data[g[0]][g[1]+1] = 255
            self.map_data[g[0]][g[1]-1] = 255
            self.map_data[g[0]-1][g[1]] = 255
            self.map_data[g[0]-1][g[1]+1] = 255
            self.map_data[g[0]-1][g[1]-1] = 255
            self.map_data[g[0]+1][g[1]] = 255
            self.map_data[g[0]+1][g[1]+1] = 255
            self.map_data[g[0]+1][g[1]-1] = 255

        resize_img = resize_grayscale_image(self.map_data, self.resize_shape)

        # save_grayscale_image(resize_img, full_path="/Users/alpha/Desktop/images/image" + str(self.count) + ".png")

        return resize_img


class SeaquestProcessor(Processor):
    def __init__(self):
        self.resize_shape = (84, 84)
        self.convert_to_grayscale = True
        self.oxy_base_line = 172
        self.oxy_detect_point = 100
        self.human_lives_base_line = 181
        self.human_pos = [63, 71, 79, 87, 95, 103]
        self.current_lives = 0
        self.human_rewards = 0
        self.is_low_oxygen = False
        self.refill_rewards = 0
        self.h_rewards = cl.deque(maxlen=100)
        self.r_rewards = cl.deque(maxlen=100)
        self.is_first = True
        self.agent_lives_base_line = 27
        self.agent_lives_detect_x = 78

    def clone(self):
        return SeaquestProcessor()

    def set_resize_shape(self, resize_shape):
        self.resize_shape = resize_shape

    def set_grayscale(self, convert_to_grayscale):
        self.convert_to_grayscale = convert_to_grayscale

    def get_grayscale(self):
        return self.convert_to_grayscale

    def get_resize_shape(self):
        return self.resize_shape

    def reset(self):
        self.is_low_oxygen = False
        self.human_rewards = 0
        self.refill_rewards = 0
        self.is_first = True
        self.current_lives = 0

    def check_data(self, img):
        # prev = self.is_low_oxygen
        # if img[self.oxy_base_line][self.oxy_detect_point] < 100:
        #     self.is_low_oxygen = True
        # else:
        #     self.is_low_oxygen = False
        # if prev is True and self.is_low_oxygen is False:
        #     if self.is_first:
        #         self.is_first = False
        #     else:
        #         self.refill_rewards = self.refill_rewards + 100
        #         self.r_rewards.append(100)

        prev = self.current_lives
        self.current_lives = 0
        for x in self.human_pos:
            if img[self.human_lives_base_line][x] < 100:
                self.current_lives = self.current_lives + 1

        if self.current_lives == prev + 1:
            self.human_rewards = self.human_rewards + 20
            self.h_rewards.append(20)

        if self.current_lives == prev - 1:
            if img[self.agent_lives_base_line][self.agent_lives_detect_x] > 100:
                # if img[self.oxy_base_line][self.oxy_detect_point] < 100:
                #     self.refill_rewards = self.refill_rewards + 100
                #     self.r_rewards.append(100)
                if prev == 6:
                    self.refill_rewards = self.refill_rewards + 1000
                    self.r_rewards.append(1000)
                else:
                    self.refill_rewards = self.refill_rewards + 100
                    self.r_rewards.append(100)

        # save_grayscale_image(img, full_path="/Users/alpha/Desktop/images/image2.png")

    def process(self, pre_image):
        gray_scale = convert_rgb_to_grayscale(pre_image)

        # save_grayscale_image(gray_scale, full_path="/home/garlicdevs/Desktop/images/image.png")

        self.check_data(gray_scale)

        resize_img = resize_grayscale_image(gray_scale, self.resize_shape)

        return resize_img

    def get_rewards(self):
        if len(self.h_rewards) > 0:
            r1 = self.h_rewards.pop()
        else:
            r1 = 0
        if len(self.r_rewards) > 0:
            r2 = self.r_rewards.pop()
        else:
            r2 = 0
        return [r1, r2]


class RiverraidProcessor(Processor):
    def __init__(self):
        self.resize_shape = (84, 84)
        self.convert_to_grayscale = True
        self.guided_map = None
        self.gas_base_line = 73
        self.gas_full = 53
        self.gas_empty = 31
        self.agent_base_line = 60
        self.frame_skip = 4
        self.total_frame = 0
        self.prev_gas = 100
        self.current_gas = 100

    def clone(self):
        return RiverraidProcessor()

    def set_resize_shape(self, resize_shape):
        self.resize_shape = resize_shape

    def set_grayscale(self, convert_to_grayscale):
        self.convert_to_grayscale = convert_to_grayscale

    def get_grayscale(self):
        return self.convert_to_grayscale

    def get_resize_shape(self):
        return self.resize_shape

    def get_guided_map(self):
        return self.guided_map

    def get_intrinsic_reward(self):
        if self.current_gas > self.prev_gas:
            return 10
        else:
            return 0

    def reset(self):
        self.prev_gas = self.current_gas = 100

    def get_gas_percentage(self):
        pos = 0
        self.prev_gas = self.current_gas
        for x in range(self.gas_empty, self.gas_full):
            if self.guided_map[self.gas_base_line][x] > 0:
                pos = x
                break
            elif self.guided_map[self.gas_base_line - 2][x] > 0:
                pos = x
                break
            elif self.guided_map[self.gas_base_line + 2][x] > 0:
                pos = x
                break
            elif self.guided_map[self.gas_base_line - 1][x] > 0:
                pos = x
                break
            elif self.guided_map[self.gas_base_line + 1][x] > 0:
                pos = x
                break
        if pos == 0:
            return -1 # Unknown
        else:
            pos = int((pos-self.gas_empty)*100/(self.gas_full-self.gas_empty))
            self.current_gas = pos
        return pos

    def process(self, pre_image):
        gray_scale = convert_rgb_to_grayscale_and_resize(pre_image, self.resize_shape)

        self.guided_map = np.copy(gray_scale)
        self.guided_map[self.guided_map < 180] = 0

        return gray_scale


class AtariBlackenProcessor(Processor):
    def __init__(self, resize_shape=(84, 84), top=0, bottom=42, left=0, right=84):
        self.resize_shape = resize_shape
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.convert_to_grayscale = True

    def set_resize_shape(self, resize_shape):
        self.resize_shape = resize_shape

    def get_resize_shape(self):
        return self.resize_shape

    def set_grayscale(self, convert_to_grayscale):
        self.convert_to_grayscale = convert_to_grayscale

    def get_grayscale(self):
        return self.convert_to_grayscale

    def clone(self):
        return AtariBlackenProcessor(self.resize_shape, self.top, self.bottom, self.left, self.right)

    def process(self, pre_image):
        if self.resize_shape is not None:
            grayscale_image = convert_rgb_to_grayscale_and_resize(pre_image, self.resize_shape)
            blacken_image(grayscale_image, self.resize_shape[0], self.resize_shape[1], self.top, self.bottom,
                          self.left, self.right)

            return grayscale_image

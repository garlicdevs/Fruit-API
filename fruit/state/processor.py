from fruit.utils.image import *


class Processor(object):
    def process(self, obj):
        pass

    def clone(self):
        pass

    def reset(self):
        pass

    def get_rewards(self, reward):
        return reward

    def get_number_of_objectives(self):
        return 1

    def get_number_of_agents(self):
        return 1


class AtariProcessor(Processor):
    def __init__(self, resize_shape=(84, 84), convert_to_grayscale=True):
        self.resize_shape = resize_shape
        self.convert_to_grayscale = convert_to_grayscale

    def set_resize_shape(self, resize_shape):
        self.resize_shape = resize_shape

    def set_grayscale(self, convert_to_grayscale):
        self.convert_to_grayscale = convert_to_grayscale

    def get_grayscale(self):
        return self.convert_to_grayscale

    def get_resize_shape(self):
        return self.resize_shape

    def process(self, pre_image):
        if self.resize_shape is not None and self.convert_to_grayscale:
            return convert_rgb_to_grayscale_and_resize(pre_image, self.resize_shape)
        elif self.convert_to_grayscale:
            return convert_rgb_to_grayscale(pre_image)
        else:
            return resize_grayscale_image(pre_image, self.resize_shape)

    def clone(self):
        return AtariProcessor(self.resize_shape, self.convert_to_grayscale)

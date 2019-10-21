from PIL import Image
import numpy as np
from fruit.process.image import imresize
import gym


def show_grayscale_image(grayscale_image):
    img = Image.fromarray(grayscale_image, 'L')
    img.show()


def show_rgb_image(rgb_image):
    img = Image.fromarray(rgb_image, 'RGB')
    img.show()


def save_grayscale_image(grayscale_image, full_path):
    img = Image.fromarray(grayscale_image, 'L')
    img.save(full_path)


def save_rgb_image(rgb_image, full_path):
    img = Image.fromarray(rgb_image, 'RGB')
    img.save(full_path)


# Nature paper uses Torch Image library image.rgb2y() to convert RGB image to grayscale,
# We will implement similarly by using Numpy. Using other library may cause side effect
# in the future updates (deprecated function, etc.)
# Ref: https://github.com/torch/image/blob/master/test/test.lua
# Input: rgb_image = (height, width, channel)
def convert_rgb_to_grayscale(rgb_image):
    return np.dot(rgb_image[:,:,:3], [0.299, 0.587, 0.114]).astype(np.uint8)


# Resize using bilinear
# Input: grayscale_image = (height, width)
#        resize_shape = (height, width)
def resize_grayscale_image(grayscale_image, resize_shape):
    return imresize(grayscale_image, resize_shape, interp='bilinear')


# Combine convert grayscale and resize to reduce error
# Input: rgb_image = (height, width, channel)
#        resize_shape = (height, width)
def convert_rgb_to_grayscale_and_resize(rgb_image, resize_shape):
    grayscale_image = np.dot(rgb_image[:, :, :3], [0.299, 0.587, 0.114])
    return imresize(grayscale_image, resize_shape, interp='bilinear')


# Blacken upper part of the image
def blacken_image(grayscale_image, width, height, top, bottom, left, right):
    for i in range(height):
        for j in range(width):
            if i > top and i < bottom and j > left and j < right:
                grayscale_image[i][j] = 0


def _unit_test_():
    
    # Test 1
    rgb = np.array([[[255, 255, 255], [32, 0, 0]], [[1, 0, 0], [0, 1, 0]]])  # 2x2x3
    print(rgb.shape)
    g = convert_rgb_to_grayscale(rgb)
    print(g)
    print(g.shape)

    # Test 2
    b = convert_rgb_to_grayscale_and_resize(rgb, (3, 3))
    print(b)
    print(b.shape)

    # Test 3
    env = gym.make("Breakout-v0")
    frame = env.reset()
    env.step(1)
    for i in range(10):
        frame, _, _, _ = env.step(0)

    grayscale = convert_rgb_to_grayscale_and_resize(frame, (84, 84))
    show_grayscale_image(grayscale)

import numba as nb
from numba import cuda
from numba.cuda import jit
import numpy as np
from PIL import Image
from image_distort import AWGN

print(cuda.gpus)

@jit(debug=True)
def MedianFilter(image_array, window_size, width_image, element_per_block_width, elements_per_block_height):
    x_pad = cuda.blockIdx.x * element_per_block_width
    y_pad = cuda.blockIdx.y * elements_per_block_height
    wind_pad = window_size // 2
    block_size = window_size * window_size
    block = cuda.local.array((49, ), nb.uint8)

    for i in range(wind_pad, elements_per_block_height + wind_pad):
        for j in range(wind_pad, element_per_block_width + wind_pad):
            for i in range(0, window_size * window_size):
                block[i] = 0
            x_true = x_pad + j - wind_pad
            y_true = y_pad + i - wind_pad
            x = x_true
            y = y_true
            i_b = 0
            i_b_max = window_size
            j_b = 0
            j_b_max = window_size
            if x == x_pad: j_b = x_pad
            elif x == element_per_block_width + x_pad: j_b = window_size - wind_pad
            if y == y_pad: i_b = y_pad
            elif y == elements_per_block_height + y_pad: i_b = window_size - wind_pad
            for z in range(i_b, i_b_max):
                for l in range(j_b, j_b_max):
                    block[(z * window_size) + l] = image_array[(y * width_image) + x]
                    x += 1
                x = x_true
                y += 1
            sorted_arr = np.sort(block, axis=None)
            image_array[(y_true * width_image) + x_true] = sorted_arr[(window_size*window_size)//2]

@cuda.jit
def power_array(a_array):
    x = cuda.blockIdx.x * 100
    for i in range(x, x + 100):
        a_array[i] *= a_array[i]

image = Image.open("baboon.png")
im = np.array(image, dtype=np.uint8)[:, :, 1]

im_d = AWGN(im, sigma=25)

Image.fromarray(im_d).show()

pixel_per_block_w = 512 // 16

blocks = (16, 16)

MedianFilter[blocks, 1](im_d, 7, 512, pixel_per_block_w, pixel_per_block_w)

Image.fromarray(im_d).show()

print("Is ok")



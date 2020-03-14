import numpy as np
from image_distort import AWGN, ASCN, Mult, Speckle, ascn2D_fft_gen
from Frost_filter import Frost
from Lee_filter import lee_filter
from Median_filter import median_filter
import PIL.Image as Image
from PIL import Image
import rasterio
from rasterio import plot
from norm import normalize_image
from dct2Dnps_add_est import dct2Dnps_add_est
import matplotlib.pyplot as plt
import time
import prepare_patch as pp
from metrics import PSNRHVSM, PSNR
from filters import dct_filter
import math
from sklearn.metrics import mean_squared_error


padsize = 5
psize = 512 + (padsize * 2)
x=512 - padsize
y=512 - padsize

jp3s = rasterio.open("T42TXR_20190313T060631_B05.jp2", driver='JP2OpenJPEG')

imarr = np.array(jp3s.read())
print(imarr.shape)

im_n = normalize_image(imarr)
im = im_n[x:x+psize, y:y+psize]
ideal_image = im[padsize:-padsize, padsize:-padsize]


result = dict()

noises = [5]
windowses = [3, 5]

for i in noises:
    noised_image = AWGN(im, dist_type="awgn", sigma=i, mu=0)
    for window_s in windowses:
        print("Median filter " + str(window_s))
        start = time.time()
        median_image = median_filter(noised_image, 9)
        finish = time.time()
        filtered_image = median_image[padsize:-padsize, padsize:-padsize]
        psnrhvs = PSNRHVSM(ideal_image, filtered_image)
        mse = mean_squared_error(ideal_image, filtered_image)
        result["Median " + str(i) + " " + str(window_s)] = [mse, PSNR(ideal_image, filtered_image, mse), psnrhvs[0], psnrhvs[1], finish - start]

    for window_s in windowses:
        print("Lee filter " + str(window_s))
        start = time.time()
        li_image = lee_filter(noised_image, 9)
        finish = time.time()
        filtered_image = li_image[padsize:-padsize, padsize:-padsize]
        psnrhvs = PSNRHVSM(ideal_image, filtered_image)
        mse = mean_squared_error(ideal_image, filtered_image)
        result["Lee " + str(i) + " " + str(window_s)] = [mse, PSNR(ideal_image, filtered_image, mse), psnrhvs[0],
                                                       psnrhvs[1], finish - start]

    for window_s in windowses:
        print("Frost filter " + str(window_s))
        start = time.time()
        frost_image = Frost(noised_image, 9)
        finish = time.time()
        filtered_image = frost_image[padsize:-padsize, padsize:-padsize]
        psnrhvs = PSNRHVSM(ideal_image, filtered_image)
        mse = mean_squared_error(ideal_image, filtered_image)
        result["Frost " + str(i) + " " + str(window_s)] = [mse, PSNR(ideal_image, filtered_image, mse), psnrhvs[0], psnrhvs[1], finish - start]

    print("DCT filt")
    start = time.time()
    dct_image = dct_filter(noised_image, i)
    finish = time.time()
    filtered_image = dct_image[padsize:-padsize, padsize:-padsize]
    psnrhvs = PSNRHVSM(ideal_image, filtered_image)
    mse = mean_squared_error(ideal_image, filtered_image)
    result["Median " + str(i) + " " + str(8)] = [mse, PSNR(ideal_image, filtered_image, mse), psnrhvs[0], psnrhvs[1], finish - start]

print(result)
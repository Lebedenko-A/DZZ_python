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
from metrics import PSNRHVSM
from filters import dct_filter
import math


psize = 512
x=3846
y=3484

jp3s = rasterio.open("/media/rostyslav/4976902B14F960E5/work/DZZ_python/T42TXQ_20190313T060631_B05.jp2", driver='JP2OpenJPEG')
plot.show(jp3s)

imarr = np.array(jp3s.read())
print(imarr.shape)

im_n = normalize_image(imarr)
im = im_n[x:x+psize, y:y+psize]
plot.show(im_n)
#start = time.time()
images = AWGN(im_n[x:x+psize, y:y+psize], dist_type="awgn", sigma=35, mu=0)

#image_ASCN = ASCN(im_n[x:x+psize, y:y+psize], nsigma=30, gsigma=1.2)
#image_Mult = Mult(im_n[x:x+psize, y:y+psize], looks=3)
#image_Speckle = Speckle(im_n[x:x+psize, y:y+psize], looks=5, gsigma=1.2)
#finish = time.time()
#Image.fromarray(image_Speckle).show()
#print(finish-start)


dctmtx = np.array([[0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
                   [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
                   [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
                   [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, 0.4157],
                   [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
                   [0.2778, -0.4904, 0.0975, 0.4175, -0.4157, -0.0975, 0.4904, -0.2778],
                   [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
                   [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0875]])

#8*8
'''dct = dct2Dnps_add_est(images, dctmtx)
print(dct)'''


tic = time.time()
m = median_filter(image_AWGN, 3)
Image.fromarray(m).show()
toc = time.time()
print(toc-tic)

#f = Frost(image, 1)
#Image.fromarray(f).show()

#im = lee_filter(image, 8, np.round(np.var(image[:]), 10))
#Image.fromarray(im).show()


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
     return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d=psnr(im_n[x:x+psize, y:y+psize],images)
print("PSNR: ", d)

summation = 0
n = im.shape
for i in range (0, n[0]):
    for j in range(0, n[1]):
      difference = im[i, j] - images[i, j]
squared_difference = difference**2
summation = summation + squared_difference
MSE = summation/n
print ("MSE: " , MSE[0])
(psnrhvs, psnrhvsm) = PSNRHVSM(im_n[x:x+psize, y:y+psize], images)
print("psnrhvs: ", psnrhvs)
print("psnrhvsm: ",psnrhvsm)
'''
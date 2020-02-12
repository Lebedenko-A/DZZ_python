import numpy as np
from image_distort import AWGN, ASCN, Mult, Speckle, ascn2D_fft_gen
from Frost_filter import Frost
from Lee_filter import lee_filter
from Median_filter import median_filter
import PIL.Image as Image
import rasterio
from rasterio import plot
from utils import normalize_image
from dct2Dnps_add_est import dct2Dnps_add_est
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import cv2


psize = 512
x=3846
y=3484

jp3s = rasterio.open("E:\\DZZ\\T42TXR_20190313T060631_B05.jp2", driver='JP2OpenJPEG')
#plot.show(jp3s)

imarr = np.array(jp3s.read())
print(imarr.shape)
im_n = normalize_image(imarr)

image = AWGN(im_n, dist_type="awgn", sigma=35, mu=0)
#image = ASCN(im_n, nsigma=30, gsigma=1.2)
#image = Mult(im_n[x:x+psize, y:y+psize], looks=3)
#image = Speckle(im_n[x:x+psize, y:y+psize], looks=5, gsigma=1.2)

Image.fromarray(image).show()

dctmtx = np.array([[0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
                   [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
                   [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
                   [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, 0.4157],
                   [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
                   [0.2778, -0.4904, 0.0975, 0.4175, -0.4157, -0.0975, 0.4904, -0.2778],
                   [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
                   [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0875]])

#8*8
dct = dct2Dnps_add_est(image[x:x+psize, y:y+psize], dctmtx)
print(dct)

#m = median_filter(image[x:x+psize, y:y+psize], 3)
#Image.fromarray(m).show()

#f = Frost(image, 1)
#Image.fromarray(f).show()

#im = lee_filter(image, 8, np.round(np.var(image[:]), 10))
#Image.fromarray(im).show()
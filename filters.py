import numpy as np
from utils.DCT import adct2, idct2
from utils.matfun import mean2d


def dct_filter(image, sigma, bsize=8, step=1):
    beta = 2.7
    s = np.shape(image)
    filtered_image = np.zeros(s, dtype=np.float)
    threshold = beta*(sigma)
    for i in range(0, s[0]-bsize, step):
        for j in range(0, s[1]-bsize, step):
            im_block = image[i:i+bsize, j:j+bsize]
            dct_block = adct2(im_block).reshape((bsize*bsize, 1))
            for z in range(1, bsize*bsize):
                if abs(dct_block[z]) <= threshold: dct_block[z] = 0
            filtered_image[i:i+bsize, j:j+bsize] = idct2(dct_block.reshape((bsize, bsize)))
    return filtered_image
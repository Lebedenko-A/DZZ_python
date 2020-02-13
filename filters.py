import numpy as np
from utils.DCT import adct2, idct2
from utils.matfun import mean2d


def dct_filter(image, sigma, bsize = 8, overflow = 0):
    beta = 2.7
    s = np.shape(image)
    filtered_image = np.zeros(s, dtype=np.float)
    step = 8
    threshold = beta*sigma
    if overflow != 0: step = 1
    for i in range(0, s[0]-bsize, step):
        for j in range(0, s[1]-bsize, step):
            block = image[i:i+bsize, j:j+bsize]
            d_block = adct2(block)
            b_thr = mean2d(d_block)*threshold
            for k in range(0, bsize):
                for l in range(0, bsize):
                    if k != 0 or l != 0:
                        if d_block[k, l] > b_thr:
                            d_block[k, l] = 0
            block = idct2(d_block)
            filtered_image[i:i+bsize, j:j+bsize] = block
    return filtered_image

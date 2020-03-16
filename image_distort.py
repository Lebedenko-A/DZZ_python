import numpy as np
import math as m
from scipy.stats import rayleigh

def AWGN(image, sigma=0, mu=0):
    im_shape = image.shape
    noise = np.random.normal(mu, sigma, im_shape)
    noised = np.ndarray.astype(image, np.float) + noise
    return noised.astype(np.uint8)


def ascn2D_fft_gen(AWGN, gsigma):
    s = AWGN.shape
    x = range(int(-s[1] / 2), int(s[1] / 2))
    y = range(int(-s[0] / 2), int(s[0] / 2))
    xgrid, ygrid = np.meshgrid(x, y)
    size = xgrid.shape
    G = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            G[i, j] = m.exp(-m.pi * (xgrid[i, j]**2 + ygrid[i, j]**2)/(2 * gsigma**2))
    g = np.fft.fft2(G)
    n = np.fft.fft2(AWGN)
    ASCN = np.fft.ifft2(g*n)
    ASCN = ASCN / np.std(ASCN)
    return ASCN


def ASCN(image, nsigma=0, gsigma=1):
    im_shape = image.shape
    nimg = image + (nsigma * ascn2D_fft_gen(np.random.randn(im_shape[0], im_shape[1]), gsigma))
    return np.uint8(nimg)

def Mult (image, looks):
    k = 0.8
    im_shape = image.shape
    noise = np.zeros(im_shape)
    for i in range(1, looks):
            noise = noise + np.random.rayleigh(k, im_shape)
    nimg = image * (noise / looks)
    return nimg.astype(np.uint8)

def Speckle(image, looks, gsigma):
    k = 0.8
    im_shape = image.shape
    noise = np.zeros(im_shape)
    for i in range(1, looks):
        ascn = ascn2D_fft_gen(np.random.randn(im_shape[0], im_shape[1]), gsigma)
        C = ascn
        size = C.shape
        B = np.random.rayleigh(k, size)
        CI = np.argsort(C)
        BI = np.argsort(B)
        C[CI] = B[BI]
        noise = noise + np.reshape(C, im_shape)
    nimg = image * (noise / looks)
    return nimg.astype(np.uint8)


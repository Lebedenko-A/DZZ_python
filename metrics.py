import numpy as np
from math import log10
from utils.DCT import adct2


CSFCof = np.array([[1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
           [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
           [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
           [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
           [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
           [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
           [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
           [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]])

MaskCof = np.array([
        [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
        [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
        [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
        [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
        [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
        [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
        [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
        [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]
])

def MSE(imQ, imP):
    s = imQ.shape
    MSE_res = float(0)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            MSE_res += pow((float(imQ[i, j]) - float(imP[i, j])), 2)

    return MSE_res/(s[0] * s[1])

def PSNR(ideal_image, noisy_image, mse):
    return 10*log10(pow(255, 2)/mse)

def PSNRHVSM(ideal_patch, noisy_patch, wstep=8):
    ip_size = ideal_patch.shape
    np_size = noisy_patch.shape
    if ip_size != np_size:
        return (0, 0)
    psnr_estimate = 0
    psnrhvsm_estimate = 0
    i, j, num = 0, 0, 0
    while i < (ip_size[0]-wstep):
        while j < (ip_size[1]-wstep):
            block_i = ideal_patch[i:i+8, j:j+8]
            block_n = noisy_patch[i:i+8, j:j+8]
            i_dct = adct2(block_i)
            n_dct = adct2(block_n)
            MaskI = maskeff(block_i, i_dct)
            MaskN = maskeff(block_n, n_dct)
            if(MaskI > MaskN):
                MaskN, MaskI = MaskI, MaskN
            j = j + wstep
            for k in range(0, 8):
                for l in range(0, 8):
                    u = abs(i_dct[k, l] - n_dct[k, l])
                    psnr_estimate = psnr_estimate + (u * CSFCof[k, l])**2
                    if k != 1 or l != 1:
                        if u < (MaskI/MaskCof[k, l]):
                            u = 0
                        else:
                            u = u - (MaskI / MaskCof[k, l])
                    psnrhvsm_estimate = psnrhvsm_estimate + (u*CSFCof[k, l])**2
                    num = num + 1
        j, i = 0, i + wstep
    if num != 0:
        psnr_estimate = psnr_estimate / num
        psnrhvsm_estimate = psnrhvsm_estimate / num
        if psnrhvsm_estimate == 0:
            psnrhvsm_estimate = 10000
        else:
            psnrhvsm_estimate = 10 * log10(255*255/psnrhvsm_estimate)
        if psnr_estimate == 0:
            psnr_estimate = 10000
        else:
            psnr_estimate = 10 * log10(255 * 255 / psnr_estimate)
    return (psnr_estimate, psnrhvsm_estimate)

def maskeff(z, zdct):
    m = 0
    for k in range(0, 8):
        for l in range(0, 8):
            m = m + (zdct[k, l]**2) * MaskCof[k, l]
    pop = vari(z)
    if pop != 0:
        pop = (vari(z[0:3, 0:3]) + vari(z[0:3, 4:7]) + vari(z[4:7, 4:7]) + vari(z[4:7, 0:3])) / pop
    m = np.sqrt(m*pop)/32
    return m

def vari(AA):
    return np.var(np.reshape(AA, AA.shape[0]*AA.shape[0]))*(AA.shape[0]*AA.shape[0])
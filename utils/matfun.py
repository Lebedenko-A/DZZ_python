import numpy as np

def mean2d(block):
    s = np.shape(block)
    if len(s) != 2:
        return 0
    mean = 0
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            mean = mean + block[i,j]
    return mean/(s[0]*s[1])

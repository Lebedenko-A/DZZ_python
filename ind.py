from PIL import Image
import numpy as np
import os
import prepare_patch as pp
from metrics import PSNRHVSM
from filters import dct_filter

dir = "/media/rostyslav/NewPart/work/ImageDatasets/TID2013/reference_images/"
list_file = os.listdir(path=dir)
print(list_file)

i = 0
while True:
    prs_name = list_file[i].split(".")
    if (prs_name[-1] == "bmp" or prs_name[-1] == "BMP"):
        im = Image.open(dir + list_file[i])
        break
    i = i + 1

first_im = np.array(im)
images = np.zeros((len(list_file), first_im.shape[0], first_im.shape[1], first_im.shape[2]))
images[0] = first_im
print(images.shape)

for i in range(1, len(list_file)):
    filename = list_file[i]
    prs_name = filename.split(".")
    if(prs_name[-1] == "bmp" or prs_name[-1] == "BMP"):
        im = Image.open(dir+filename)
        np_im = np.array(im)
        images[i] = np_im


(patches, patch_size) = pp.cut_images(images, 32)
print(patches.shape)

noise_patch = pp.add_AWGN_to_patch(patches, np.array([0.001, 0.01, 0.1, 1]))
print(noise_patch[0:patch_size].shape)

(psnr, psnrhvsm) = PSNRHVSM(patches[0:10], noise_patch[0:10])

print(psnr)
print(psnrhvsm)


filtered_blocks = np.zeros((10,32,32))
for i in range(0, 10):
    filtered_blocks[i] = dct_filter(noise_patch[i], 0.001)

(psnr, psnrhvsm) = PSNRHVSM(patches[0:10], filtered_blocks)
print(psnr)
print(psnrhvsm)

(psnr, psnrhvsm) = PSNRHVSM(filtered_blocks, noise_patch[0:10])
print(psnr)
print(psnrhvsm)

'''
im = Image.fromarray(patches[2])
im.show()
'''
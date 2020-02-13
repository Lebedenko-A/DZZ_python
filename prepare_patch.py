import numpy as np

def cut_images(images, block_size):
    #   return: patch from image with size (block_size, block_size)
    #   parameters: images - images array with size (image_count, image_width, image_height, channels)
    #               block_size
    k = images.shape
    patch_size = 1
    for i in range(0, len(k)):
        if(i == 1 or i == 2):
            patch_size = patch_size * int(k[i]/(block_size/2))
        else:
            patch_size = patch_size * k[i]
    patches = np.zeros((patch_size, block_size, block_size), dtype=np.uint8)
    patches_pointer = 0
    for i in range(0, k[0]):
        for j in range(0, k[3]):
            for x in range(0, k[1]-int(block_size/2), int(block_size/2)):
                for y in range(0, k[2]-int(block_size/2), int(block_size/2)):
                    if patches_pointer < patch_size:
                        patches[patches_pointer] = images[i, x:x+block_size, y:y+block_size, j]
                        patches_pointer = patches_pointer + 1

    return (patches, patch_size)

def add_AWGN_to_patch(image_patch, noise_sigma):
    ns_size = np.shape(noise_sigma)
    ip_size = np.shape(image_patch)
    noise_patch = np.zeros((ip_size[0]*ns_size[0], ip_size[1], ip_size[2]), dtype=np.float)
    np_pointer = 0
    for i in noise_sigma:
        for j in (0, ip_size[0]-1):
            noise = np.random.normal(0, i, (ip_size[1], ip_size[2]))
            noise_patch[np_pointer] = image_patch[j] + noise
            np_pointer = np_pointer+1
    return noise_patch
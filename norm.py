import numpy as np

def normalize_image(image_array, min=0, max=255):
    s = np.shape(image_array)
    linear_image = np.reshape(image_array, (s[1]*s[2], 1))
    minimum_im = np.min(linear_image)
    maximum_im = np.max(linear_image)

    normal_image = (linear_image/(maximum_im - minimum_im)) * max
    return np.reshape(normal_image.astype(np.uint8), (s[1], s[2]))
import os
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve

color_map = 'gray'
gauss_sigma = 1.8
threshold = 255 * 0.85  # 85% of 8-bit gray-scaled image


def apply_laplacian(images):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    for i, image in enumerate(images):
        image = gaussian_filter(image, sigma=gauss_sigma)
        image = convolve(image, kernel)
        image = np.where(image >= threshold, np.max(image), 0)
        images[i] = image

    return images


directory = 'C:\\Users\\lukas\\OneDrive\\Desktop\\Programming\\' \
            'PycharmProjects\\EdgeDetectionPaper\\images\\fingerprints'

image_list = list()
for root, _, files in os.walk(directory):
    for file in files:
        image_list.append(PIL.Image.open(os.path.join(root, file)))

image_list = apply_laplacian(image_list)

plt.imshow(image_list[0], cmap='gray')
plt.show()

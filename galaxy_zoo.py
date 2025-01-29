import zipfile
from matplotlib import image as mpimg
import numpy as np


# with zipfile.ZipFile("gz2_images_all.zip", "r") as zip_ref:
#     zip_ref.extractall("Galaxy Zoo Images")






image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/project/Galaxy Zoo Images/gz2_images_all/587739379910508635.jpg")

smallest_non_zero = np.min(image[image > 0])
image = np.where(image == 0.0, smallest_non_zero, image)

print(image.shape)
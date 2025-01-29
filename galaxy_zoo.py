import zipfile
from matplotlib import image as mpimg
import numpy as np


# with zipfile.ZipFile("gz2_images_all.zip", "r") as zip_ref:
#     zip_ref.extractall("Galaxy Zoo Images")



# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T


image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/project/Galaxy Zoo Images/gz2_images_all/587739379910508635.jpg")



print(np.min(image))
print(np.max(image))

image = normalise_independently(image)

print(np.min(image))
print(np.max(image))
print(image.shape)

if len(image[0] < 256):
    # enlarge (stretch) the image to 256x256 with bicubic interpolation (best for enlarging images although slower than bilinear)
    image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_CUBIC)
else:
    # shrink the image to 256x256 using area interpolation (best for shrinking images)
    image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_AREA)

print(image.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import keras
from keras import layers
from astropy.io import fits
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

images = []


hdu_list_1 = np.array(fits.open("Images/ui.3016979176.1_0.fits"))
hdu_list_2 = np.array(fits.open("Images/ui.3016990447.1_0.fits"))
images.append(hdu_list_1[0].data)
images.append(hdu_list_2[0].data)
images = np.array(images)

print(images.shape)

images = np.expand_dims(images, axis=3)

print(images.shape)

max_pool_images = MaxPooling2D(pool_size=2, padding="same")(images)
upscaled_images = UpSampling2D(size=2)(max_pool_images)

print(images.shape)
print(max_pool_images.shape)
print(upscaled_images.shape)

plt.figure(figsize=(15,10))

ax_2 = plt.subplot(2, 3, 1)
plt.imshow(np.array(images[0]).reshape(50, 50))

ax_2 = plt.subplot(2, 3, 2)
plt.imshow(np.array(max_pool_images[0]).reshape(25, 25))

ax_2 = plt.subplot(2, 3, 3)
plt.imshow(np.array(upscaled_images[0]).reshape(50, 50))

ax_2 = plt.subplot(2, 3, 4)
plt.imshow(np.array(images[1]).reshape(50, 50))

ax_2 = plt.subplot(2, 3, 5)
plt.imshow(np.array(max_pool_images[1]).reshape(25, 25))

ax_2 = plt.subplot(2, 3, 6)
plt.imshow(np.array(upscaled_images[1]).reshape(50, 50))


plt.show()
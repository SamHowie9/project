import pandas as pd
from astropy.io import fits
from keras.src.ops import shape
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt
import cv2


images = ["broadband_540856.fits", "broadband_546348.fits", "broadband_166270.fits", "broadband_247336.fits", "broadband_391637.fits", "broadband_592000.fits", "broadband_600893.fits", "broadband_89584.fits", "broadband_589571.fits", "broadband_204076.fits"]

fig, axs = plt.subplots(len(images), 6, figsize=(10, 20))


for i in range(0, len(images)):
    for j in range(0, 5):
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)

axs[0][0].set_title("Original")
axs[0][1].set_title("Gaussian Filter")
axs[0][2].set_title("Gaussian Noise")
axs[0][3].set_title("Normalisation")
axs[0][4].set_title("Resizing")


for j, image_name in enumerate(images):

    # hdu_list = fits.open("sdss/snapnum_095/data/broadband_540856.fits")
    hdu_list = fits.open("sdss/snapnum_095/data/" + image_name)
    image = hdu_list[0].data


    # take only the g,r,i bands (ignore z)
    image = image[0:3]


    axs[j][0].imshow(((np.array(image)/np.max(image)).T))




    # convert fwhm value to sigma
    def fwhm_to_sigma(fwhm):
        return fwhm/(2 * np.sqrt(2 * np.log(2)))

    # apply gaussian filter to each band
    image[0] = gaussian_filter(image[0], sigma=fwhm_to_sigma(1.5))
    image[1] = gaussian_filter(image[1], sigma=fwhm_to_sigma(1.5))
    image[2] = gaussian_filter(image[2], sigma=fwhm_to_sigma(2))


    axs[j][1].imshow((np.array(image)/np.max(image)).T)




    # add random gaussian noise to each band
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.1, (len(image[0]), len(image[0])))
        image[i] = image[i] + gaussian


    axs[j][2].imshow((np.array(image)/np.max(image)).T)




    # normalise each band
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i] - np.min(image[i])))


    axs[j][3].imshow(np.array(image).T)




    # convert image to numpy array of type float32 (for the cv2 resizing function to work)
    image = np.array(image).astype(np.float32)

    # image resizing (enlarging and shrinking use different interpolation algorithms for the best results
    if len(image[0] < 256):
        # enlarge (stretch) the image to 256x256 with bicubic interpolation (best for enlarging images although slower than bilinear)
        image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_CUBIC)
    else:
        # shrink the image to 256x256 using area interpolation (best for shrinking images)
        image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_AREA)


    axs[j][4].imshow(image)


    # normalise each band
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i] - np.min(image[i])))


    axs[j][5].imshow(np.array(image).T)



fig.tight_layout()
plt.savefig("Variational TNG/Plots/Image Processing")
plt.show()

from astropy.io import fits
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt


images = ["broadband_166270.fits", "broadband_247336.fits", "broadband_304313.fits", "broadband_391637.fits", "broadband_540856.fits", "broadband_546348.fits"]

fig, axs = plt.subplots(len(images), 5, figsize=(10, 10))


for i in range(0, len(images)):
    for j in range(0, 5):
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)

axs[0][0].set_title("Original")
axs[0][1].set_title("Gaussian Filter")
axs[0][2].set_title("Gaussian Noise")
axs[0][3].set_title("Normalisation")
axs[0][4].set_title("Log Filter")


for j, image_name in enumerate(images):

    # hdu_list = fits.open("sdss/snapnum_095/data/broadband_540856.fits")
    hdu_list = fits.open("sdss/snapnum_095/data/" + image_name)
    image = hdu_list[0].data
    image = image[0:3]

    image = np.int32(np.array(image).round())

    print(np.array(image).shape)

    print(image[0].max())
    print(image[1].max())
    print(image[2].max())

    axs[j][0].imshow(np.array(image).T)


    def fwhm_to_sigma(fwhm):
        return fwhm/(2 * np.sqrt(2 * np.log(2)))

    image[0] = gaussian_filter(image[0], sigma=fwhm_to_sigma(1.5))
    image[1] = gaussian_filter(image[1], sigma=fwhm_to_sigma(1.5))
    image[2] = gaussian_filter(image[2], sigma=fwhm_to_sigma(2))


    axs[j][1].imshow(np.array(image).T)


    gaussian = np.random.normal(0, 0.1, (len(image[0]), len(image[0])))

    for i in range(0, 3):
        image[i] = image[i] + gaussian


    axs[j][2].imshow(np.array(image).T)

    # normalisation and log filter
    for i in range(0, 3):
        image[i] = image[i]/image[i].max()
        # image[i] = np.log10(image[i]) + 1

    axs[j][3].imshow(np.array(image).T)


    for i in range(0, 3):
        image[i] = np.log10(image[i]) + 1

    image = np.float32(image)

    axs[j][4].imshow(np.array(image).T)

fig.tight_layout()
plt.savefig("Variational TNG/Plots/Image Processing")
plt.show()

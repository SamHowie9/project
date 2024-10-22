from astropy.io import fits
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt

hdu_list = fits.open("sdss/snapnum_095/data/broadband_540856.fits")
image = hdu_list[0].data
image = image[0:3]

fig, axs = plt.subplots(1, 3)

axs[0].imshow(np.array(image).T)


def fwhm_to_sigma(fwhm):
    return fwhm/np.sqrt((8 * np.log(2)))

image[0] = gaussian_filter(image[0], sigma=fwhm_to_sigma(1.5))
image[1] = gaussian_filter(image[1], sigma=fwhm_to_sigma(1.5))
image[2] = gaussian_filter(image[2], sigma=fwhm_to_sigma(2))

axs[1].imshow(np.array(image).T)

# normalisation and log filter
for i in range(0, 3):
    image[i] = image[i]/image[i].max()
    # image[i] = np.log10(image[i]) + 1

axs[2].imshow(np.array(image).T)

plt.show()

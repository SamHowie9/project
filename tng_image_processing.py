from astropy.io import fits
from scipy import gaussian_filter
import numpy as np


hdu_list = fits.open("sdss/snapnum_095/data/broadband_540856.fits")
image = hdu_list[0].data
image = image[0:3]

# normalisation and log filter
for i in range(0, 3):
    image[i] = image[i]/image[i].max()
    image[i] = np.log10(image[i]) + 1


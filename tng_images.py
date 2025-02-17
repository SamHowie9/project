import h5py
import tarfile
from astropy.io import fits
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import os
from scipy.ndimage import gaussian_filter
import cv2



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# file = tarfile.open("/cosma7/data/durham/dc-howi1/project/TNG100/sdss_095.tar")
#
# for member in file.getmembers():
#     if member.name.startswith("sdss/snapnum_095/data/"):
#         # print(member.name)
#         file.extract(member=member, path="/cosma7/data/durham/dc-howi1/project/TNG Images/")

    # if member.isdir():
    #     print(member.name)

    # if not(member.name.startswith("sdss/snapnum_095/data/broadband") or member.name.startswith("sdss/snapnum_095/morph_images_i/") or member.name.startswith("sdss/snapnum_095/morph_images_g/")):
    #     print(member.name)

# file.extractall(path="sdss/snapnum_095/data/")

# file.extract("sdss/snapnum_095/data/broadband_540856.fits")
# file.extract("sdss/snapnum_095/data/broadband_546348.fits")
# file.extract("sdss/snapnum_095/data/broadband_166270.fits")
# file.extract("sdss/snapnum_095/data/broadband_247336.fits")
# file.extract("sdss/snapnum_095/data/broadband_391637.fits")
# file.extract("sdss/snapnum_095/data/broadband_592000.fits")
# file.extract("sdss/snapnum_095/data/broadband_600893.fits")
# file.extract("sdss/snapnum_095/data/broadband_89584.fits")
# file.extract("sdss/snapnum_095/data/broadband_589571.fits")
# file.extract("sdss/snapnum_095/data/broadband_204076.fits")

# file.extract("sdss/snapnum_095/morphs_i.hdf5")
# file.extract("sdss/snapnum_095/morphs_g.hdf5")



# hdu_list_1 = fits.open("sdss/snapnum_095/data/broadband_540856.fits")
# image_1 = hdu_list_1[0].data
# # image_1 = np.array(image_1[0:3]).T
#
# hdu_list_2 = fits.open("sdss/snapnum_095/data/broadband_546348.fits")
# image_2 = hdu_list_2[0].data
# # image_2 = np.array(image_2[0:3]).T
#
# hdu_list_3 = fits.open("sdss/snapnum_095/data/broadband_166270.fits")
# image_3 = hdu_list_3[0].data
# # image_3 = np.array(image_3[0:3]).T
#
# hdu_list_4 = fits.open("sdss/snapnum_095/data/broadband_247336.fits")
# image_4 = hdu_list_4[0].data
# # image_4 = np.array(image_4[0:3]).T
#
# hdu_list_5 = fits.open("sdss/snapnum_095/data/broadband_391637.fits")
# image_5 = hdu_list_5[0].data
# # image_5 = np.array(image_5[0:3]).T
#
# for i in range(0, 4):
#     image_1[i] = image_1[i]/image_1[i].max()
#     image_2[i] = image_2[i]/image_2[i].max()
#     image_3[i] = image_3[i]/image_3[i].max()
#     image_4[i] = image_4[i]/image_4[i].max()
#     image_5[i] = image_5[i]/image_5[i].max()
#
#     image_1[i] = np.log10(image_1[i]) + 1
#     image_2[i] = np.log10(image_2[i]) + 1
#     image_3[i] = np.log10(image_3[i]) + 1
#     image_4[i] = np.log10(image_4[i]) + 1
#     image_5[i] = np.log10(image_5[i]) + 1
#
#     print(image_1[i].max())
#
# print(image_1.max())
#
# image_1 = np.array(image_1[0:3]).T
# image_2 = np.array(image_2[0:3]).T
# image_3 = np.array(image_3[0:3]).T
# image_4 = np.array(image_4[0:3]).T
# image_5 = np.array(image_5[0:3]).T
#
#
#
# # df = pd.DataFrame(np.array(hdu_list_1[0].data)[0])
# # print(df)
#
#
#
# fig, axs = plt.subplots(1, 5, figsize=(25, 5))
#
# axs[0].imshow(image_1)
# axs[1].imshow(image_2)
# axs[2].imshow(image_3)
# axs[3].imshow(image_4)
# axs[4].imshow(image_5)
#
# for i in range(0, 5):
#     axs[i].get_yaxis().set_visible(False)
#     axs[i].get_xaxis().set_visible(False)
#
#
#
#
# # hdu_list = fits.open("sdss/snapnum_095/data/broadband_540856.fits")
# # image = hdu_list[0].data
# #
# # for i in range(0, 4):
# #     image[i] = image[i]/image[i].max()
# #
# # print(image.max())
# #
# # image = np.array(image[0:3]).T
# # plt.imshow(image)
#
#
# # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# # # axs[0].imshow(image[0]/image[0].max())
# # axs[0].imshow(image[0])
# # axs[1].imshow(image[1])
# # axs[2].imshow(image[2])
# # axs[3].imshow(image[3])
# #
# # for i in range(0, 4):
# #     axs[i].get_xaxis().set_visible(False)
# #     axs[i].get_yaxis().set_visible(False)
#
# plt.savefig("Variational TNG/Plots/tng_test")
# plt.show()








# file.extractall()



# f = h5py.File("/cosma7/data/durham/dc-howi1/project/TNG100/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_091.hdf5", "r")
#
# # print(f.keys())
#
# indices = np.array(f["subhaloIDs"])
#
# print(indices)

# images = pd.DataFrame(f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"])

# indices = np.array(f)
#
# images = np.array(f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"])
#
# face_on = np.transpose(np.transpose(images)[0])
#
# first_image_all_bands = face_on[0]
#
# first_image_ugr = first_image_all_bands[0:3]
#
#
# print(images.shape)
# print(face_on.shape)
# print(first_image_all_bands)
# print(first_image_ugr.shape)


# print(np.array(f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"]).shape)
# print(np.array(f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"])[0].shape)
# print(np.array(f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"])[0][0].shape)

# print(images)

# print(list(f.keys()))







# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T

# normalise each band to r
def normalise_to_r(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[1]) - np.min(image[1]))
    return image.T

# convert fwhm value to sigma
def fwhm_to_sigma(fwhm):
    return fwhm/(2 * np.sqrt(2 * np.log(2)))

# image processing for each image
def image_processing(image):

    # take only the g,r,i bands (ignore z)
    image = image[0:3]

    # apply gaussian filter to each band
    image[0] = gaussian_filter(image[0], sigma=fwhm_to_sigma(1.5))
    image[1] = gaussian_filter(image[1], sigma=fwhm_to_sigma(1.5))
    image[2] = gaussian_filter(image[2], sigma=fwhm_to_sigma(2))

    # random seed for reproducibility
    random.seed(1)

    # add random gaussian noise to each band of the image
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.1, (len(image[0]), len(image[0])))
        image[i] = image[i] + gaussian

    # convert image to numpy array of type float32 (for the cv2 resizing function to work)
    image = np.array(image).astype(np.float32)

    # image resizing (enlarging and shrinking use different interpolation algorithms for the best results
    if len(image[0] < 256):
        # enlarge (stretch) the image to 256x256 with bicubic interpolation (best for enlarging images although slower than bilinear)
        image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_CUBIC)
    else:
        # shrink the image to 256x256 using area interpolation (best for shrinking images)
        image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_AREA)

    #transpose for normalisation
    image = image.T

    # normalise each band individually
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i] - np.min(image[i])))

    # return the new image
    return image.T







fig, axs = plt.subplots(5, 5, figsize=(15, 15))

# list of all the galaxy images to load
galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/TNG Images/sdss/snapnum_095/data/")

random.seed(1)
random_galaxies = random.sample(galaxies, 25)

n = 0

for i in range(0, 5):
    for j in range(0, 5):

        hdu_list = fits.open("/cosma7/data/durham/dc-howi1/project/TNG Images/sdss/snapnum_095/data/" + random_galaxies[n])
        image = hdu_list[0].data

        image = image_processing(image)

        # # perform the image processing
        # image = image_processing(image)

        axs[i][j].imshow(image)
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)

        n += 1

plt.savefig("Variational TNG/Plots/random_galaxy_sample", bbox_inches='tight')
plt.show()
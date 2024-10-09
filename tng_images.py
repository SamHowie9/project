import h5py
import tarfile
from astropy.io import fits
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# file = tarfile.open("/cosma7/data/durham/dc-howi1/project/TNG100/sdss_095.tar")
#
# # for member in file.getmembers():
# #     if member.name.startswith("sdss/snapnum_095/data/"):
# #         print(member.name)
#
# # file.extractall(path="sdss/snapnum_095/data/")
#
# file.extract("sdss/snapnum_095/data/broadband_540856.fits")
# file.extract("sdss/snapnum_095/data/broadband_546348.fits")
# file.extract("sdss/snapnum_095/data/broadband_166270.fits")
# file.extract("sdss/snapnum_095/data/broadband_247336.fits")
# file.extract("sdss/snapnum_095/data/broadband_391637.fits")



hdu_list_1 = fits.open("sdss/snapnum_095/data/broadband_540856.fits")
image_1 = hdu_list_1[0].data
image_1 = np.array(image_1[0:3]).T

hdu_list_2 = fits.open("sdss/snapnum_095/data/broadband_546348.fits")
image_2 = hdu_list_2[0].data
image_2 = np.array(image_2[0:3]).T

hdu_list_3 = fits.open("sdss/snapnum_095/data/broadband_166270.fits")
image_3 = hdu_list_3[0].data
image_3 = np.array(image_3[0:3]).T

hdu_list_4 = fits.open("sdss/snapnum_095/data/broadband_247336.fits")
image_4 = hdu_list_4[0].data
image_4 = np.array(image_4[0:3]).T

hdu_list_5 = fits.open("sdss/snapnum_095/data/broadband_391637.fits")
image_5 = hdu_list_5[0].data
image_5 = np.array(image_5[0:3]).T


df = pd.DataFrame(np.array(hdu_list_1[0].data)[0])
print(df)


# plt.imshow(image)

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
# # fig, axs = plt.subplots(4, 1)
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
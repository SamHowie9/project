import h5py
import tarfile
import numpy as np
import pandas as pd


file = tarfile.open("/cosma7/data/durham/dc-howi1/project/TNG100/sdss_095.tar")

for member in file.getmembers():
    if member.isdir():
        print(member.name)

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
import h5py
import numpy as np
import pandas as pd

f = h5py.File("/cosma7/data/durham/dc-howi1/project/TNG100/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_091.hdf5", "r")

images = pd.DataFrame(f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"])

print(images)

# print(list(f.keys()))
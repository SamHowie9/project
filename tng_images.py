import h5py

f = h5py.File("~/TNG100/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_091.hdf5", "r")

print(list(f.keys()))
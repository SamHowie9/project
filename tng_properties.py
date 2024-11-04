import h5py
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import requests
import eagleSqlTools as sql


headers = {"api-key":"dca7ae0d7be7a9186da1a9d6b2536a78"}
snapshot_url = "http://www.tng-project.org/api/TNG100-1/snapshots/67/subhalos/"
group_catalogue_url = "http://www.tng-project.org/api/TNG100-1/snapshots/67/subhalos/info.json"
group_catalogue_url = "http://www.tng-project.org/api/TNG100-1/files/groupcat-67/"

group_catalogue = requests.get(group_catalogue_url, headers=headers).json()

print(group_catalogue.keys())

print(group_catalogue["mass_stars"])

# print(group_catalogue.text)

# a = group_catalogue.json()

# print(type(group_catalogue))

# print(group_catalogue["url"])

# print(group_catalogue.keys())

# subhalos = requests.get(snapshot_url, headers=headers).json()

# stellar_mass = []
#
# print(subhalos["count"])
#
# # for i in range(subhalos["count"]):
# for i in range(5):
#
#     subhalo_url = snapshot_url + f"{i}/"
#     subhalo = requests.get(subhalo_url, headers=headers).json()
#
#     stellar_mass.append((subhalo["mass_stars"]))
#
#     # print(subhalo.keys())
#
# print(stellar_mass)
#
# # np.save()










# data_g = h5py.File("sdss/snapnum_095/morphs_g.hdf5")
# data_i = h5py.File("sdss/snapnum_095/morphs_i.hdf5")
#
# print(list(data_g.keys()))
# print(list(data_i.keys()))
#
# # print(list(data_g["sersic_n"]))
#
# sersic_g = list(data_g["sersic_n"])
# sersic_i = list(data_i["sersic_n"])
#
# print(len(sersic_g), len(sersic_i))
#
# i = 0
# while i < len(sersic_g):
#     if sersic_g[i] > 12 or sersic_g[i] < 0 or sersic_i[i] > 12 or sersic_g[i] < 0:
#         sersic_g.pop(i)
#         sersic_i.pop(i)
#         i-=1
#     i+=1
#
# print(len(sersic_g), len(sersic_i))
#
# plt.scatter(sersic_g, sersic_i, alpha=0.1)
# plt.show()

# plt.scatter(list(data_g["sersic_n"]), list(data_i["sersic_n"]))
# plt.show()
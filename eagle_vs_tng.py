import numpy as np
from matplotlib import pyplot as plt
import requests
import eagleSqlTools as sql










# headers = {"api-key":"dca7ae0d7be7a9186da1a9d6b2536a78"}
# base_url = "http://www.tng-project.org/api/"
# path = base_url + "TNG100-1/snapshots/"
#
# # all_snapshots = requests.get(path, headers=headers).json()
# #
# # # print(all_snapshots.keys())
# #
# # print(all_snapshots)
# #
# #
# # snapshot_url = all_snapshots[99]["url"]
# # catalog_url = snapshot_url + "subhalos/"
# # subhalos = requests.get(catalog_url).json()
#
#
# snapshot_url = "http://www.tng-project.org/api/TNG100-1/snapshots/67/subhalos/"
#
# subhalos = requests.get(snapshot_url, headers=headers).json()
#
# stellar_mass = []
#
# # for i in range(subhalos["count"]):
# for i in range(1):
#
#     subhalo_url = snapshot_url + f"{i}/"
#     subhalo = requests.get(subhalo_url, headers=headers).json()
#
#     stellar_mass.append((subhalo["mass_stars"]))
#
#     # print(subhalo.keys())
#
# np.save()
#
#
# # print(catalog_url)
#
# # print(data.keys())
#
#
#
#
#
#
# # baseUrl = "http://www.tng-project.org/api/"
# #
# # path = baseUrl + "TNG100-1/"
# # # path = baseUrl
# #
# # # r = requests.get(path, params=None, headers=headers)
# # # data = r.json()
#
#
#
# # print(data["mass_dm"])
#
# # print(type(r))
#
# # print(data.keys())
#
# # print(data["num_snapshots"])
#
# # for i in range(len(data['simulations'])):
# #     print(data['simulations'][i])
#
# # stellar_masses = data["mass_stars"]
# #
# # print(stellar_masses)
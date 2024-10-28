from matplotlib import pyplot as plt
import requests
import eagleSqlTools as sql


baseUrl = "http://www.tng-project.org/api/"
headers = {"api-key":"dca7ae0d7be7a9186da1a9d6b2536a78"}

path = baseUrl + "TNG100-1/"
# path = baseUrl

r = requests.get(path, params=None, headers=headers)
data = r.json()

print(data["mass_dm"])

# print(type(r))

# print(data.keys())

# for i in range(len(data['simulations'])):
#     print(data['simulations'][i])

# stellar_masses = data["mass_stars"]
#
# print(stellar_masses)
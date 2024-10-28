from matplotlib import pyplot as plt
import requests
import eagleSqlTools as sql


baseUrl = "http://www.tng-project.org/api/"
headers = {"api-key":"dca7ae0d7be7a9186da1a9d6b2536a78"}

r = requests.get(baseUrl, params=None, headers=headers)
data = r.json()

print(data.keys())

# stellar_masses = data["mass_stars"]
#
# print(stellar_masses)
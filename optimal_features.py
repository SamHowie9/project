import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

extracted_features = np.load("Features/32_features.npy")

df1 = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")

structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")

print(df1["GalaxyID"].tolist())
print(df1)
print(structure_properties)

plt.figure(figsize=(8,8))

plt.scatter(y=structure_properties["re_r"], x=structure_properties["re_star"], s=1)

plt.xlim(0, 120)
plt.ylim(0, 120)

plt.ylabel("re_r")
plt.xlabel("re_star")
plt.title("Semi-Major Axis")

plt.savefig("Plots/semi-major_full")
plt.show()

# structual features: Sersic index, Axis ratio, position angle, semi-major axis

# physical propeties: stellar mass, star formation rate, halo mass, black hole mass, merger history

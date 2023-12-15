from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd
# import seaborn as sns




# load the supplemental file into a dataframe
df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
df.drop(df.tail(200).index, inplace=True)


# all_images = []
#
# # loop through each galaxy in the supplemental file
# for i, galaxy in enumerate(df["GalaxyID"].tolist()):
#
#     # get the filename for that galaxy
#     filename = "galface_" + str(galaxy) + ".png"
#
#     # open the image and append it to the main list
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
#     all_images.append(image)




# fig, axs = plt.subplots(2, 10, figsize=(20, 8))




# print(df)
# load the data
df_ab = pd.read_csv("Galaxy Properties/absolute_magnitudes.csv", comment="#")

# print(df_ab)

df_ab = df.merge(df_ab, how="left", on="GalaxyID")
df_ab = df_ab.dropna()

# print(df_ab)

df_ab = df_ab[["GalaxyID", "r_nodust"]]
# df = df[["GalaxyID", "g_nodust", "r_nodust", "i_nodust"]]

# print(df_ab)

magnitudes = [-23, -22.5, -22, -21.5, -21.25, -21, -20.75, -20.5, -20, -19.5]

galaxies = []

for i in magnitudes:
    closest_mag = df_ab.iloc[(df_ab["r_nodust"]-i).abs().argsort()[0]]
    closest_galaxy = str(int(closest_mag["GalaxyID"].tolist()))
    galaxies.append(closest_galaxy)


chosen_images = []

for galaxy in galaxies:
    # get the filename for that galaxy
    filename = "galface_" + galaxy + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    chosen_images.append(image)

chosen_images = np.array(chosen_images)

print(chosen_images.shape)
print(chosen_images[0].shape)



fig, axs = plt.subplots(2, 10, figsize=(20, 8))

for i in range(len(galaxies)):
    axs[0, i].imshow(galaxies[i])


plt.savefig("Plots/absolute_mag_images")







# fig, axs = plt.subplots(1, 4, figsize=(20, 8))
#
# sns.histplot(ax=axs[0], data=df, x="g_nodust", element="poly")
# sns.histplot(data=df_ab, x="r_nodust", element="poly", bins=50)
# sns.histplot(ax=axs[0], data=df, x="i_nodust", element="poly")
#
# sns.histplot(ax=axs[1], data=df, x="g_nodust", element="poly")
# sns.histplot(ax=axs[2], data=df, x="r_nodust", element="poly")
# sns.histplot(ax=axs[3], data=df, x="i_nodust", element="poly")

# sns.scatterplot(data=df, x="g_nodust", y="r_nodust", hue="i_nodust")

plt.show()
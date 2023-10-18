from astropy.io import fits
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# create a list to store all of the image data to train the model
train_images = []

# file = "ui.3007920713.1_0"

hdu_list = fits.open("Images/ui.3007920713.1_0.fits")
image_data = hdu_list[0].data
# print(image_data)
df = pd.DataFrame(image_data)
print(df)
# train_images.append(image_data)

# for file in os.listdir("Images"):
#     hdu_list = fits.open("Images/" + file)
#     image_data = hdu_list[0].data
#     train_images.append(image_data)
#
# print(train_images)


# # file = "ui.3007920713.1_0"
# # path = "Images/" + file
# # print(path)
#
# hdu_list = fits.open("Images/desY1stripe82_GZ1_ES.fits")
# # hdu_list.info()
# # print(hdu_list[1].data)
#
# df = pd.DataFrame(hdu_list[1].data)
# # print(df[df["SPIRAL"] == 0])
# # print(df[["filename", "SPIRAL", "ELLIPTICAL", "UNCERTAIN"]])
# # print(df[df["SPIRAL"] == 1])
#
# galaxy_types = []
# df2 = df[["SPIRAL", "ELLIPTICAL", "UNCERTAIN"]]
# # df2["galaxy_type"] = df2.apply(lambda x: str(x.name[x.ne(0)]), axis=1)
# # df2["galaxy_type"] = df2.where(df == 0, other=df.apply(lambda x: x.name), axis=1).where(df != 0, other="").apply(lambda row: ''.join(row.values), axis=1)
# df["galaxy_type"] = df[["SPIRAL", "ELLIPTICAL", "UNCERTAIN"]].where(df == 0, other=df.apply(lambda x: x.name), axis=1).where(df != 0, other="").apply(lambda row: ''.join(row.values), axis=1)
#
# print(df[["filename", "SPIRAL", "ELLIPTICAL", "UNCERTAIN", "galaxy_type"]])
#
# galaxy_types = df["galaxy_type"].to_list()
#
# s=0
# e=0
# u=0
#
# for type in galaxy_types:
#     if type == "SPIRAL":
#         s+=1
#     elif type == "ELLIPTICAL":
#         e+=1
#     elif type ++ "UNCERTAIN":
#         u+=1
#     else:
#         print(".")
#
# print(s, e, u, (s+e))
#
#
# # galaxy_types = df2["galaxy_type"].to_numpy()
# # np.set_printoptions(threshold = np.inf)
# # print(galaxy_types)
#
#
# # image_data = hdu_list[0].data
# # print(type(image_data))
# # print(image_data)
# # plt.imshow(image_data)
#
# # plt.show()
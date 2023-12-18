from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd
# import seaborn as sns
import cv2
import random




# load the supplemental file into a dataframe
df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
df.drop(df.tail(200).index, inplace=True)


# sns.histplot(data=df, x="galfit_mag")


# magnitudes = [15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19]
#
# galaxies = []
#
# for i in magnitudes:
#     closest_mag = df.iloc[(df["galfit_mag"]-i).abs().argsort()[0]]
#     closest_galaxy = str(int(closest_mag["GalaxyID"].tolist()))
#     galaxies.append(closest_galaxy)
#
# print(galaxies)



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




# # print(df)
# # load the data
# df_ab = pd.read_csv("Galaxy Properties/absolute_magnitudes.csv", comment="#")
#
# # print(df_ab)
#
# df_ab = df.merge(df_ab, how="left", on="GalaxyID")
# df_ab = df_ab.dropna()
#
# # print(df_ab)
#
# df_ab = df_ab[["GalaxyID", "r_nodust"]]
# # df = df[["GalaxyID", "g_nodust", "r_nodust", "i_nodust"]]
#
# # print(df_ab)
#
# magnitudes = [-23, -22.5, -22, -21.5, -21.25, -21, -20.75, -20.5, -20, -19.5]
#
# galaxies = []
#
# for i in magnitudes:
#     closest_mag = df_ab.iloc[(df_ab["r_nodust"]-i).abs().argsort()[0]]
#     closest_galaxy = str(int(closest_mag["GalaxyID"].tolist()))
#     galaxies.append(closest_galaxy)


# get a list of 12 random indices for each group
random_index = random.sample(range(0, len(df)), 12)

# get the galaxy id of each of the random indices for each group
galaxies = df["GalaxyID"].iloc[random_index].tolist()

galaxies[3] = 2065457
galaxies[7] = 5341887

chosen_images = []

for galaxy in galaxies:
    # get the filename for that galaxy
    filename = "galface_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

    chosen_images.append(image)

chosen_images = np.array(chosen_images)



def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img




def resize_image(image, cutoff):

    mean_intensity = image.mean()

    intensity_x = image.mean(axis=2).mean(axis=0)
    intensity_y = image.mean(axis=2).mean(axis=1)

    half_max_intensity_x = np.max(intensity_x)/2
    half_max_intensity_y = np.max(intensity_y)/2

    size = len(intensity_x)

    start_x = 0
    start_y = 0
    end_x = 255
    end_y = 255

    found_start_x = False
    found_start_y = False
    found_end_x = False
    found_end_y = False

    # loop through half of the image
    for j in range(0, int(size/2)):

        # if we haven't previously found the cutoff point and are still below the cutoff, increment the pointer
        if found_start_x is False and intensity_x[j]/mean_intensity < half_max_intensity_x:
            start_x += 1
        else:
            found_start_x = True

        if found_end_x is False and intensity_x[j]/mean_intensity < half_max_intensity_x:
            end_x -= 1
        else:
            found_end_x = True

        if found_start_y is False and intensity_y[j]/mean_intensity < half_max_intensity_y:
            start_x += 1
        else:
            found_start_y = True

        if found_end_y is False and intensity_y[j]/mean_intensity < half_max_intensity_y:
            end_y -= 1
        else:
            found_end_y = True

    return start_x, end_x

    # for j in range(0, int(size/2)):
    #
    #     if (intensity_x[j] > cutoff) and (found_start_x == 0):
    #         start_x = j
    #         found_start_x = 1
    #
    #     if (intensity_x[-j] > cutoff) and (found_end_x == 0):
    #         end_x = 255 - j
    #         found_end_x = 1
    #
    #     if (intensity_y[j] > cutoff) and (found_start_y == 0):
    #         start_y = j
    #         found_start_y = 1
    #
    #     if (intensity_y[-j] > cutoff) and (found_end_y == 0):
    #         end_y = 255 - j
    #         found_end_y = 1
    #
    #
    # # check if image is too large to crop, if no we have to scale it down to 128, 128
    # if start_x < 64 and start_y < 64 and end_x > 192 and end_y > 192:
    #     image = cv2.resize(image, (128, 128))
    #
    # # if the image isn't too large, we can do a center crop
    # else:
    #     image = center_crop(image, (128, 128))
    #
    # print()
    # return image







fig, axs = plt.subplots(3, 12, figsize=(35, 10))

for i in range(len(galaxies)):

    axs[0, i].imshow(chosen_images[i])
    axs[0, i].axvline(x=64, c="white")
    axs[0, i].axvline(x=192, c="white")
    axs[0, i].axhline(y=64, c="white")
    axs[0, i].axhline(y=192, c="white")
    axs[0, i].get_xaxis().set_visible(False)
    axs[0, i].get_yaxis().set_visible(False)

    # image = resize_image(image=chosen_images[i], cutoff=0.075)
    # axs[1, i].imshow(image)
    # axs[1, i].get_xaxis().set_visible(False)
    # axs[1, i].get_yaxis().set_visible(False)

    # image = resize_image(image=chosen_images[i], cutoff=0.06)
    # axs[2, i].imshow(image)

    # cutoff = 0.075
    #
    #

    mean_intensity = image.mean()

    intensity_x = image.mean(axis=2).mean(axis=0)
    intensity_y = image.mean(axis=2).mean(axis=1)

    half_max_intensity_x = np.max(intensity_x) / 2
    half_max_intensity_y = np.max(intensity_y) / 2

    x_min, x_max = resize_image(chosen_images[i], cutoff=0.075)

    # # intensity_x = chosen_images[i].mean(axis=2).mean(axis=0)
    # axs[1, i].bar(x=range(0, len(intensity_x)), height=intensity_x, width=1)
    # # axs[1, i].axvline(x=64, c="black")
    # # axs[1, i].axvline(x=192, c="black")
    # # axs[1, i].axhline(y=0.06, c="black", alpha=0.2)
    # axs[1, i].get_xaxis().set_visible(False)
    # # axs[1, i].set_ylim([0, 0.35])

    axs[1, i].bar(x=range(0, len(intensity_x)), height=intensity_x/mean_intensity, width=1)
    # axs[2, i].axvline(x=64, c="black")
    # axs[2, i].axvline(x=192, c="black")
    # axs[2, i].axhline(y=0.06, c="black", alpha=0.2)
    axs[1, i].get_xaxis().set_visible(False)
    axs[1, i].axvspan(x_min, x_max, facecolor="yellow", alpha=0.2)
    axs[1, i].axvline(x=64, c="black")
    axs[1, i].axvline(x=192, c="black")
    # axs[2, i].set_ylim([0, 0.35])


    # intensity_y = chosen_images[i].mean(axis=2).mean(axis=1)
    # axs[2, i].barh(y=range(0, len(intensity_y)), width=intensity_y, height=1)
    # axs[2, i].axhline(y=64, c="black")
    # axs[2, i].axhline(y=192, c="black")
    # axs[2, i].axvline(x=0.06, c="black", alpha=0.2)
    # axs[2, i].get_yaxis().set_visible(False)
    # axs[2, i].set_xlim([0, 0.35])

    # start_x = 0
    # start_y = 0
    # end_x = 255
    # end_y = 255
    # found_start_x = False
    # found_start_y = False
    # found_end_x = False
    # found_end_y = False
    #
    # for j in range(0, 129):
    #     if intensity_x[j] > cutoff:
    #         start_x = j
    #         break
    # for j in range(255, 127, -1):
    #     if intensity_x[j] > cutoff:
    #         end_x = j
    #         break
    # for j in range(0, 129):
    #     if intensity_y[j] > cutoff:
    #         start_y = j
    #         break
    # for j in range(255, 127, -1):
    #     if intensity_y[j] > cutoff:
    #         end_y = j
    #         break
    #
    # print(start_x, end_x)
    # print(start_y, end_y)
    # print()
    #
    # axs[0, i].axvline(x=start_x, c="yellow")
    # axs[0, i].axvline(x=end_x, c="yellow")
    #
    # axs[0, i].axhline(y=start_y, c="yellow")
    # axs[0, i].axhline(y=end_y, c="yellow")



plt.savefig("Plots/resizing_images")





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
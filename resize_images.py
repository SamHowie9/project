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







# # get a list of 12 random indices for each group
# random_index = random.sample(range(0, len(df)), 12)
#
# # get the galaxy id of each of the random indices for each group
# galaxies = df["GalaxyID"].iloc[random_index].tolist()
#
# galaxies[3] = 2065457
# galaxies[7] = 5341887
#
# # galaxies = [9793595, 16696731, 16238798, 2065457, 9279688, 13681352, 10138699, 5341887, 14949191, 9231886, 8132671, 13174674]
#
# galaxies = [13681352, 2065457, 14949191]
#
# print(galaxies)
#
# chosen_images = []
#
# for galaxy in galaxies:
#     # get the filename for that galaxy
#     filename = "galface_" + str(galaxy) + ".png"
#
#     # open the image and append it to the main list
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
#
#     chosen_images.append(image)
#
# chosen_images = np.array(chosen_images)





# crop to the center of the image
def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img



# find the half maximum points
def half_max_range(image):

    mean_intensity = image.mean()

    intensity_x = image.mean(axis=2).mean(axis=0)
    intensity_y = image.mean(axis=2).mean(axis=1)

    half_max_intensity_x = np.max(intensity_x/mean_intensity) / 2
    half_max_intensity_y = np.max(intensity_y/mean_intensity) / 2

    # print()
    # print(half_max_intensity_x, half_max_intensity_y)

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
    for j in range(0, int(size / 2)):


        # if we haven't previously found the cutoff point and are still below the cutoff, increment the pointer
        if (found_start_x is False) and ((intensity_x[j] / mean_intensity) < half_max_intensity_x):
            start_x += 1
        else:
            found_start_x = True

        if (found_end_x is False) and ((intensity_x[-j] / mean_intensity) < half_max_intensity_x):
            end_x -= 1
        else:
            found_end_x = True

        if (found_start_y is False) and ((intensity_y[j] / mean_intensity) < half_max_intensity_y):
            start_y += 1
        else:
            found_start_y = True

        if (found_end_y is False) and ((intensity_y[-j] / mean_intensity) < half_max_intensity_y):
            end_y -= 1
        else:
            found_end_y = True

    # print(start_x, end_x, start_y, end_y)
    return start_x, end_x, start_y, end_y


# resizes the galaxy images (crop or scale)
def resize_image(image, cutoff=60):

    # get the fill width half maximum (for x and y direction)
    start_x, end_x, start_y, end_y = half_max_range(image)

    # calculate the full width half maximum
    range_x = end_x - start_x
    range_y = end_y - start_y

    # check if the majority of out image is within the cutoff range, if so, center crop, otherwise, scale image down
    if (range_x <= cutoff) and (range_y <= cutoff):
        image = center_crop(image, (128, 128))
    else:
        image = cv2.resize(image, (128, 128))

    # return the resized image
    return image






all_images = []
fwhm_x = []
fwhm_y = []

# loop through each galaxy in the supplemental file
for i, galaxy in enumerate(df["GalaxyID"].tolist()):

    # get the filename for that galaxy
    filename = "galface_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    all_images.append(image)

    # get the fill width half maximum (for x and y direction)
    start_x, end_x, start_y, end_y = half_max_range(image)

    # calculate the full width half maximum
    range_x = end_x - start_x
    range_y = end_y - start_y

    fwhm_x.append(range_x)
    fwhm_y.append(range_y)

print(fwhm_x)
print()
print()
print(fwhm_y)



# fig, axs = plt.subplots(1, 2, figsize=(20, 10))
#
# sns.histplot(ax=axs[0], x=range_x)
# sns.histplot(ax=axs[1], x=range_y)
#
# plt.savefig("Plots/fwhm_distribution")



# fig, axs = plt.subplots(3, 3, figsize=(20, 20))
#
# for i in range(0, 3):
#
#     # display the original image
#     axs[0, i].imshow(chosen_images[i])
#     axs[0, i].get_xaxis().set_visible(False)
#     axs[0, i].get_yaxis().set_visible(False)
#
#     # find the mean intensity and mean intensity along the x axis for that image
#     mean_intensity = chosen_images[i].mean()
#     intensity_x = chosen_images[i].mean(axis=2).mean(axis=0)
#
#     # get the cutoff points for that image
#     x_min, x_max, y_min, y_max = half_max_range(chosen_images[i])
#
#     # plot intensity as a ratio of mean intensity
#     axs[1, i].bar(x=range(0, len(intensity_x)), height=(intensity_x/mean_intensity), width=1)
#     axs[1, i].axvspan(x_min, x_max, facecolor="yellow", alpha=0.5)
#     # axs[1, i].axvline(x=98, c="black")
#     # axs[1, i].axvline(x=158, c="black")
#
#     # display the resizd image
#     axs[2, i].imshow(resize_image(chosen_images[i]))
#     axs[2, i].get_xaxis().set_visible(False)
#     axs[2, i].get_yaxis().set_visible(False)

# fig, axs = plt.subplots(2, 3, figsize=(20, 15))
#
# for i in range(0, 3):
#
#     # display the original image
#     axs[0, i].imshow(chosen_images[i])
#     axs[0, i].get_xaxis().set_visible(False)
#     axs[0, i].get_yaxis().set_visible(False)
#
#     # find the mean intensity and mean intensity along the x axis for that image
#     mean_intensity = chosen_images[i].mean()
#     intensity_x = chosen_images[i].mean(axis=2).mean(axis=0)
#
#     # get the cutoff points for that image
#     x_min, x_max, y_min, y_max = half_max_range(chosen_images[i])
#
#     # plot intensity as a ratio of mean intensity
#     axs[1, i].bar(x=range(0, len(intensity_x)), height=(intensity_x/mean_intensity), width=1)
#     axs[1, i].axvspan(x_min, x_max, facecolor="yellow", alpha=0.5)
#     axs[1, i].axvline(x=98, c="black")
#     axs[1, i].axvline(x=158, c="black")
#
#     # display the resizd image
#     # axs[2, i].imshow(resize_image(chosen_images[i]))
#     # axs[2, i].get_xaxis().set_visible(False)
#     # axs[2, i].get_yaxis().set_visible(False)
#
# plt.savefig("Plots/resize_image_demo_full")
# plt.show()



# fig, axs = plt.subplots(3, 12, figsize=(35, 10))

# for i in range(len(galaxies)):
#
#     axs[0, i].imshow(chosen_images[i])
#     axs[0, i].axvline(x=64, c="white")
#     axs[0, i].axvline(x=192, c="white")
#     axs[0, i].axhline(y=64, c="white")
#     axs[0, i].axhline(y=192, c="white")
#     axs[0, i].get_xaxis().set_visible(False)
#     axs[0, i].get_yaxis().set_visible(False)
#
#
#     axs[1, i].imshow(resize_image(chosen_images[i]))
#     axs[1, i].get_xaxis().set_visible(False)
#     axs[1, i].get_yaxis().set_visible(False)
#
#
#     # mean_intensity = chosen_images[i].mean()
#     #
#     # intensity_x = chosen_images[i].mean(axis=2).mean(axis=0)
#     # intensity_y = chosen_images[i].mean(axis=2).mean(axis=1)
#
#     # half_max_intensity_x = np.max(intensity_x) / 3
#     # half_max_intensity_y = np.max(intensity_y) / 3
#
#     # x_min, x_max, y_min, y_max = half_max_range(chosen_images[i])
#
#     # # intensity_x = chosen_images[i].mean(axis=2).mean(axis=0)
#     # axs[1, i].bar(x=range(0, len(intensity_x)), height=intensity_x, width=1)
#     # # axs[1, i].axvline(x=64, c="black")
#     # # axs[1, i].axvline(x=192, c="black")
#     # # axs[1, i].axhline(y=0.06, c="black", alpha=0.2)
#     # axs[1, i].get_xaxis().set_visible(False)
#     # # axs[1, i].set_ylim([0, 0.35])
#     #
#     # axs[1, i].bar(x=range(0, len(intensity_x)), height=(intensity_x/mean_intensity), width=1)
#     # # axs[2, i].axvline(x=64, c="black")
#     # # axs[2, i].axvline(x=192, c="black")
#     # # axs[2, i].axhline(y=0.06, c="black", alpha=0.2)
#     # axs[1, i].get_xaxis().set_visible(False)
#     # axs[1, i].axvspan(x_min, x_max, facecolor="yellow", alpha=0.5)
#     # axs[1, i].axvline(x=64, c="black")
#     # axs[1, i].axvline(x=192, c="black")
#     #
#     #
#     # intensity_y = chosen_images[i].mean(axis=2).mean(axis=1)
#     # axs[2, i].barh(y=range(0, len(intensity_y)), width=(intensity_y/mean_intensity), height=1)
#     # axs[2, i].axhspan(y_min, y_max, facecolor="yellow", alpha=0.5)
#     # axs[2, i].axhline(y=64, c="black")
#     # axs[2, i].axhline(y=192, c="black")
#     # axs[2, i].axvline(x=0.06, c="black", alpha=0.2)
#     # axs[2, i].get_yaxis().set_visible(False)
#
# plt.savefig("Plots/resizing_images")





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

# plt.show()
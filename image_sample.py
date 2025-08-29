from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pandas as pd
import numpy as np
from skimage import color





# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T





all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")



# # galaxies = [10189421, 16140911, 13667604, 16770171, 8497741, 10412989, 18026173, 17710821, 16559175, 15934507, 17528947, 12648880, 17528947, 15528598, 9006563, 10138699, 17995800, 9389474, 10891574, 9393167, 9768388, 9027313, 15851558, 10237264, 10362672]
# # galaxies = [2641811, 15362842, 16914856, 56099343, 13985849, 6084349, 19099219, 14402768, 2641811, 14402768, 18981609, 25470675, 54410350, 3540463, 21623615, 21355632, 2641811, 56099343, 14237115, 62973912, 4580964, 21355632, 1028772, 10586238, 56099343]
# galaxies = [6659447, 12704761, 10202983, 138061, 10222240, 15435358, 10515382, 9731857, 8557264, 10479082, 9923144, 8316347, 9344085, 18349471, 9627417, 9715326, 8883149, 17568884, 13654434, 8503366, 8404920, 10421754, 15898377, 14974620, 8650032]
#
#
# fig, axs = plt.subplots(5, 5, figsize=(25, 25))
#
# n = 0
# for i in range(0, 5):
#     for j in range(0, 5):
#
#         galaxy = galaxies[n]
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxy) + ".png")
#
#         axs[i][j].imshow(image)
#         axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#         dt = all_properties[all_properties["GalaxyID"] == galaxy]["DiscToTotal"].values[0]
#         sersic = all_properties[all_properties["GalaxyID"] == galaxy]["n_r"].values[0]
#         axs[i][j].set_title("D/T=" + str(round(dt, 3)) + ", n=" + str(round(sersic, 3)), fontsize=20)
#
#         n += 1
#
# plt.savefig("Variational Eagle/2D Visualisation/sersic_sample_3", bbox_inches="tight")





# galaxies = [16360007, 16360007, 12099013, 14188900, 7196586, 15797866, 13255951, 16214875, 15494458]
# galaxies = [19927839, 17086648, 16360007, 20574262, 19927839, 10672055, 12159163, 19076567, 13965937]
# galaxies = [10007497, 10506037, 9432094, 13806265, 9264862, 10007497, 9672292, 11079170, 9248510]
# galaxies = [9010268, 8349164, 10953355, 13296286, 17691609, 9954916, 11140837, 12109008, 10178759]
# galaxies = [2637010, 2637010, 9714040, 9714040, 15927500, 10637654, 13885346, 10520697, 3537188]
galaxies = [10835614, 17511062, 10835614, 14059470, 19891778, 15418092, 19891778, 10835614, 13246030]

fig, axs = plt.subplots(3, 3, figsize=(25, 25))

n = 0
for i in range(0, 3):
    for j in range(0, 3):

        galaxy = galaxies[n]
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")

        # axs[i][j].imshow(image)
        # axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

        image = normalise_independently(image)
        image = color.rgb2gray(image)

        axs[i][j].imshow(image, cmap="gray_r")
        axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

        dt = all_properties[all_properties["GalaxyID"] == galaxy]["DiscToTotal"].values[0]
        sersic = all_properties[all_properties["GalaxyID"] == galaxy]["n_r"].values[0]
        axs[i][j].set_title("D/T=" + str(round(dt, 3)) + ", n=" + str(round(sersic, 3)), fontsize=20)

        n += 1

plt.savefig("Variational Eagle/2D Visualisation/sample_reversed_6", bbox_inches="tight")

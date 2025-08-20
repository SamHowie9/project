from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pandas as pd


all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")



galaxies = [10189421, 16140911, 13667604, 16770171, 8497741, 10412989, 18026173, 17710821, 16559175, 15934507, 17528947, 12648880, 17528947, 15528598, 9006563, 10138699, 17995800, 9389474, 10891574, 9393167, 9768388, 9027313, 15851558, 10237264, 10362672]

fig, axs = plt.subplots(5, 5, figsize=(25, 25))

n = 0
for i in range(0, 5):
    for j in range(0, 5):

        galaxy = galaxies[n]
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxy) + ".png")

        axs[i][j].imshow(image)

        dt = all_properties[all_properties["GalaxyID"] == galaxy]["DiscToTotal"].values[0]
        sersic = all_properties[all_properties["n_r"] == galaxy]["DiscToTotal"].values[0]
        axs[0][i].set_title("D/T=" + str(round(dt, 3) + ", n=" + str(round(sersic, 3))))

        n += 1

plt.savefig("Variational Eagle/2D Visualisation/sersic_sample_1", bbox_inches="tight")

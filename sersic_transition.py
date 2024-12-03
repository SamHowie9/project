import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random



# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")



# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
print(bad_fit)

# remove those galaxies
for i, galaxy in enumerate(bad_fit):
    # extracted_features = np.delete(extracted_features, galaxy-i, 0)
    all_properties = all_properties.drop(galaxy, axis=0)



# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"] < 1]), 4))
# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(1, 2, inclusive="left")]), 4))
# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(2, 3, inclusive="left")]), 4))
# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(3, 4, inclusive="left")]), 4))
# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(4, 5, inclusive="left")]), 4))
# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(5, 6, inclusive="left")]), 4))
# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(6, 7, inclusive="left")]), 4))
# print(random.sample(list(all_properties["GalaxyID"].loc[all_properties["n_r"] >= 7]), 4))


sersic_0_1 = [2599936, 17944813, 8494197, 8485275]
sersic_1_2 = [16027782, 9489651, 8837359, 9098658]
sersic_2_3 = [14829299, 9214339, 9971165, 10696181]
sersic_3_4 = [9292390, 9192917, 14892360, 14111615]
sersic_4_5 = [9019265, 18002123, 10056399, 9052812]
sersic_5_6 = [8065154, 12575320, 16701964, 17672887]
sersic_6_7 = [9526568, 16204628, 14105654, 14747624]
sersic_7_8 = [14715402, 16657911, 15250310, 15274006]


fig = plt.figure(constrained_layout=False, figsize=(15, 8))

# create sub figures within main figure, specify their location
gs1 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.025, right=0.225, top=0.975, bottom=0.775)
gs2 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.275, right=0.475, top=0.975, bottom=0.775)
gs3 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.525, right=0.725, top=0.975, bottom=0.775)
gs4 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.775, right=0.975, top=0.975, bottom=0.775)

gs5 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.025, right=0.225, top=0.725, bottom=0.525)
gs6 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.275, right=0.475, top=0.725, bottom=0.525)
gs7 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.525, right=0.725, top=0.725, bottom=0.525)
gs8 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.775, right=0.975, top=0.725, bottom=0.525)

count = 0

for i in range(0, 2):
    for j in range(0, 2):

        g1_ax = fig.add_subplot(gs1[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_0_1[count]) + ".png")
        g1_ax.imshow(image)
        g1_ax.get_xaxis().set_visible(False)
        g1_ax.get_xaxis().set_visible(False)

        g2_ax = fig.add_subplot(gs2[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_1_2[count]) + ".png")
        g2_ax.imshow(image)
        g2_ax.get_yaxis().set_visible(False)
        g2_ax.get_xaxis().set_visible(False)
        g2_ax.get_xaxis().set_visible(False)

        g3_ax = fig.add_subplot(gs3[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_2_3[count]) + ".png")
        g3_ax.imshow(image)
        g3_ax.get_xaxis().set_visible(False)
        g3_ax.get_xaxis().set_visible(False)

        g4_ax = fig.add_subplot(gs4[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_3_4[count]) + ".png")
        g4_ax.imshow(image)
        g4_ax.get_xaxis().set_visible(False)
        g4_ax.get_xaxis().set_visible(False)

        g5_ax = fig.add_subplot(gs5[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_4_5[count]) + ".png")
        g5_ax.imshow(image)
        g5_ax.get_xaxis().set_visible(False)
        g5_ax.get_yaxis().set_visible(False)

        g6_ax = fig.add_subplot(gs6[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_5_6[count]) + ".png")
        g6_ax.imshow(image)
        g6_ax.get_xaxis().set_visible(False)
        g6_ax.get_yaxis().set_visible(False)

        g7_ax = fig.add_subplot(gs7[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_6_7[count]) + ".png")
        g7_ax.imshow(image)
        g7_ax.get_xaxis().set_visible(False)
        g7_ax.get_yaxis().set_visible(False)

        g8_ax = fig.add_subplot(gs8[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(sersic_7_8[count]) + ".png")
        g8_ax.imshow(image)
        g8_ax.get_xaxis().set_visible(False)
        g8_ax.get_yaxis().set_visible(False)


        count += 1


ax1 = fig.add_subplot(gs1[:])
ax1.axis("off")
ax1.set_title("$n < 1$", fontsize=25)

ax2 = fig.add_subplot(gs2[:])
ax2.axis("off")
ax2.set_title("1 \leq n < 2", fontsize=25)

ax3 = fig.add_subplot(gs3[:])
ax3.axis("off")
ax3.set_title("2 \leq n < 3", fontsize=25)

ax4 = fig.add_subplot(gs4[:])
ax4.axis("off")
ax4.set_title("3 \leq n < 4", fontsize=25)

ax5 = fig.add_subplot(gs5[:])
ax5.axis("off")
ax5.set_title("4 \leq n < 5", fontsize=25)

ax6 = fig.add_subplot(gs6[:])
ax6.axis("off")
ax6.set_title("6 \leq n < 6", fontsize=25)

ax7 = fig.add_subplot(gs7[:])
ax7.axis("off")
ax7.set_title("6 \leq n < 7", fontsize=25)

ax8 = fig.add_subplot(gs8[:])
ax8.axis("off")
ax8.set_title("n \geq 7", fontsize=25)


plt.savefig("Variational Eagle/Plots/sersic_transition_plot", bbox_inches='tight')
plt.show()



# plt.hist(all_properties["n_r"], bins=20)


plt.show()
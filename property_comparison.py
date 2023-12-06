import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
import pandas as pd
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import os
from sklearn.cluster import KMeans



# # load the extracted features
extracted_features = np.load("Features/9_features_new.npy")


extracted_features = np.flipud(np.rot90(extracted_features))

print(extracted_features.shape)

# load the two excel files into dataframes
df1 = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
df2 = pd.read_csv("stab3510_supplemental_file/table2.csv", comment="#")



print(df1.shape)
print(df2.shape)

# loop through each galaxy in the supplmental file
for i, galaxy in enumerate(df1["GalaxyID"].tolist()):

    filename = "galface_" + str(galaxy) + ".png"

    if filename not in os.listdir("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/"):
        df1.drop(axis=0, index=i, inplace=True)
        df2.drop(axis=0, index=i, inplace=True)


print(df1.shape)
print(df2.shape)



ab_magnitude = df1["galfit_mag"]
mass = df2["galfit_lmstar"]
semi_major = (df1["galfit_re"] + df2["galfit_re"]) / 2
sersic = (df1["galfit_n"] + df2["galfit_n"]) / 2
axis_ratio = (df1["galfit_q"] + df2["galfit_q"]) / 2
position_angle = (df1["galfit_PA"] + df2["galfit_PA"]) / 2




# load the extracted features
extracted_features = np.load("Features/9_features_new.npy")

# k means clustering for the extracted features
kmeans = KMeans(n_clusters=9, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(extracted_features)


# extracted_feature_df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"])
# extracted_feature_df["Category"] = clusters


# print(df)
#
# kws = dict(s=5, linewidth=0)
#
# sns.pairplot(df, corner=True, hue="Category", plot_kws=kws, palette="colorblind")
#
# plt.savefig("Plots/9_feature_clustering_new")
# plt.show()





fig, axs = plt.subplots(6, 9, figsize=(25,20))

for i in range(0, 9):

    title = "Feature " + str(i+1)
    axs[0, i].set_title(title)

    axs[0, i].scatter(extracted_features[i], ab_magnitude, s=2, hue=clusters)
    axs[1, i].scatter(extracted_features[i], mass, s=2, hue=clusters)
    axs[2, i].scatter(extracted_features[i], semi_major, s=2, hue=clusters)
    axs[3, i].scatter(extracted_features[i], sersic, s=2, hue=clusters)
    axs[4, i].scatter(extracted_features[i], axis_ratio, s=2, hue=clusters)
    axs[5, i].scatter(extracted_features[i], position_angle, s=2, hue=clusters)

axs[0, 0].set_ylabel("AB Magnitude")
axs[1, 0].set_ylabel("Stellar Mass")
axs[2, 0].set_ylabel("Semi-Major Axis")
axs[3, 0].set_ylabel("Sersic Index")
axs[4, 0].set_ylabel("Axis Ratio")
axs[5, 0].set_ylabel("Position Angle")


plt.savefig("Plots/9_feature_property_comparison_clusters")
plt.show()



# df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"])
#
#
# # create the pairplot with custom marker size
# kws = dict(s=10)
# g = sns.pairplot(df, corner=True, plot_kws=kws)
#
#
# # function to add the correlation coefficient to the plots
# def corrfunc(x, y, ax=None, color=None):
#     # find the correlation coefficient and round to 3 dp
#     correlation = np.corrcoef(x, y)[0][1]
#     correlation = np.round(correlation, decimals=3)
#
#     # annotate the plot with the correlation coefficient
#     ax = ax or plt.gca()
#     ax.annotate(("ρ = " + str(correlation)), xy=(0.1, 1), xycoords=ax.transAxes)
#
#
# # add the correlation coefficient to each of the scatter plots
# g.map_lower(corrfunc)
#
# # add some vertical space between the plots (given we are adding the correlation coefficients
# plt.subplots_adjust(hspace=0.2)
#
#
# plt.savefig("Plots/9_feature_histogram_new")
# plt.show()

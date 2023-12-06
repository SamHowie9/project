import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import seaborn as sns
from sklearn.cluster import KMeans


# set the encoding dimension (number of extracted features)
encoding_dim = 9



# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features_new.npy")



# # stores an empty list to contain all the image data to train the model
# all_images = []
#
# # load the supplemental file into a dataframe
# df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
#
# print(df.shape)
#
# # loop through each galaxy in the supplmental file
# for i, galaxy in enumerate(df["GalaxyID"].tolist()):
#
#     filename = "galface_" + str(galaxy) + ".png"
#
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
#     all_images.append(image)
#
#
# print(np.array(all_images).shape)






kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(extracted_features)



print(clusters)
print(clusters.shape)




# centers = kmeans.cluster_centers_
# # print(centers)
#
# for center in centers:
#     print(center)
#
# # print(centers[0])
#
# reconstructions = decoder.predict(centers)
#
# print(reconstructions[0])
#
# # create figure to hold subplots
# fig, axs = plt.subplots(2, 5, figsize=(20,10))
#
# axs[0, 0].imshow(reconstructions[0])
# axs[0, 0].get_xaxis().set_visible(False)
# axs[0, 0].get_yaxis().set_visible(False)
#
# axs[0, 1].imshow(reconstructions[1])
# axs[0, 1].get_xaxis().set_visible(False)
# axs[0, 1].get_yaxis().set_visible(False)
#
# axs[0, 2].imshow(reconstructions[2])
# axs[0, 2].get_xaxis().set_visible(False)
# axs[0, 2].get_yaxis().set_visible(False)
#
# axs[0, 3].imshow(reconstructions[3])
# axs[0, 3].get_xaxis().set_visible(False)
# axs[0, 3].get_yaxis().set_visible(False)
#
# axs[0, 4].imshow(reconstructions[4])
# axs[0, 4].get_xaxis().set_visible(False)
# axs[0, 4].get_yaxis().set_visible(False)
#
# axs[1, 0].imshow(reconstructions[5])
# axs[1, 0].get_xaxis().set_visible(False)
# axs[1, 0].get_yaxis().set_visible(False)
#
# axs[1, 1].imshow(reconstructions[6])
# axs[1, 1].get_xaxis().set_visible(False)
# axs[1, 1].get_yaxis().set_visible(False)
#
# axs[1, 2].imshow(reconstructions[7])
# axs[1, 2].get_xaxis().set_visible(False)
# axs[1, 2].get_yaxis().set_visible(False)
#
# axs[1, 3].imshow(reconstructions[8])
# axs[1, 3].get_xaxis().set_visible(False)
# axs[1, 3].get_yaxis().set_visible(False)
#
# axs[1, 4].set_axis_off()
#
#
# plt.savefig("Plots/9_cluster_reconstruction_new")
# plt.show()




df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"])
df["Category"] = clusters


# print(df)

kws = dict(s=5, linewidth=0)

sns.pairplot(df, corner=True, hue="Category", plot_kws=kws, palette="colorblind")

plt.savefig("Plots/9_feature_clustering_new")
plt.show()


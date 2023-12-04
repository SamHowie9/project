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




# load the extracted features
extracted_features = np.load("Features/7_features.npy")

kmeans = KMeans(n_clusters=7, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(extracted_features)

centers = kmeans.cluster_centers_
# print(centers)

for center in centers:
    print(center)


df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7"])
df["Category"] = clusters


# print(df)

# kws = dict(s=5, linewidth=0)
#
# sns.pairplot(df, corner=True, hue="Category", plot_kws=kws, palette="colorblind")
#
# plt.savefig("Plots/7_feature_clustering")
# plt.show()


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




# open the file to load the extracted features
f = open("Features/6_features.txt", "r")
extracted_features = np.array(eval(f.read()))


kmeans = KMeans(n_clusters=6, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(extracted_features)

print(clusters)

# lists to store the values of each image for each extracted feature
f1 = []
f2 = []
f3 = []
f4 = []
f5 = []
f6 = []

# loop through each pair of values for each image and add the values to the individual lists
for i in range(extracted_features.shape[0]):
    f1.append(extracted_features[i][0])
    f2.append(extracted_features[i][1])
    f3.append(extracted_features[i][2])
    f4.append(extracted_features[i][3])
    f5.append(extracted_features[i][4])
    f6.append(extracted_features[i][5])


df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6"])
df["Category"] = clusters


print(df)

kws = dict(s=5, linewidth=0)

sns.pairplot(df, corner=True, hue="Category", plot_kws=kws, palette="colorblind")

plt.savefig("Plots/6_feature_clustering")
plt.show()


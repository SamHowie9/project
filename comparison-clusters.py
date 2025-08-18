from xml.sax.handler import all_properties

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import NearestCentroid






pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


np.set_printoptions(linewidth=np.inf)



# 16, 25

run = 2
encoding_dim = 35
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32



all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")


for run in [2]:

    # load the extracted features
    extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
    print(extracted_features.shape)

    # perform pca on the extracted features
    pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
    extracted_features = pca.transform(extracted_features)
    print(extracted_features.shape)


    # hdb = HDBSCAN().fit(extracted_features)
    # clusters = hdb.labels_
    # print(clusters)
    # print(set(clusters))
    # print(np.sum(clusters==0))
    # print(np.sum(clusters==1))
    # print(np.sum(clusters==2))
    # print(np.sum(clusters==-1))
    # print()

    kmeans = KMeans(n_clusters=11, random_state=0).fit(extracted_features)
    clusters = kmeans.labels_

    all_properties["cluster"] = clusters

    print(all_properties)

    sersic = []

    for i in range(0, 11):

        cluster_sersic = all_properties[all_properties["cluster"] == i]["n_r"].tolist()

        sersic.append([i, min(cluster_sersic), np.median(cluster_sersic), max(cluster_sersic)])

        print(sersic[i])

        # min_sersic.append(min(sersic))
        # med_sersic.append(np.median(sersic))
        # max_sersic.append(max(sersic))

    # sort by the median value
    sersic = sorted(sersic, key=lambda x: x[2])
    sersic = np.array(sersic)

    for a in sersic: print(a)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # plt.plot(sersic.T[1])
    # plt.plot(sersic.T[2])
    # plt.plot(sersic.T[3])

    order = all_properties.groupby("cluster")["n_r"].mean().sort_values().index

    sns.boxplot(ax=axs, data=all_properties, x="cluster", y="n_r", order=order)

    plt.show()






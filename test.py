import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random
# import cv2
# import keras
# from keras import ops
# from tensorflow.keras import backend as K
# import tensorflow as tf


features = np.load("Variational Eagle/Extracted Features/Fully Balanced/15_feature_750_epoch_32_bs_features_1.npy")[0]

print(features.shape)

fig, axs = plt.subplots(3, 5, figsize=(15, 12))

for i in range(0, 5):
    print(i)

    mean_1 = np.round(np.mean(features.T[i]), 2)
    mean_2 = np.round(np.mean(features.T[i+5]), 2)
    mean_3 = np.round(np.mean(features.T[i+10]), 2)

    std_1 = np.round(np.std(features.T[i]), 2)
    std_2 = np.round(np.std(features.T[i+5]), 2)
    std_3 = np.round(np.std(features.T[i+10]), 2)


    sns.histplot(ax=axs[0][i], x=features.T[i])
    sns.histplot(ax=axs[1][i], x=features.T[i+5])
    sns.histplot(ax=axs[2][i], x=features.T[i+10])

    axs[0][i].set_title("μ=" + str(mean_1) + ", σ=" + str(std_1))
    axs[1][i].set_title("μ=" + str(mean_2) + ", σ=" + str(std_2))
    axs[2][i].set_title("μ=" + str(mean_3) + ", σ=" + str(std_3))

    axs[0][i].set_yticks([])
    axs[1][i].set_yticks([])
    axs[2][i].set_yticks([])

    axs[0][i].set_yticklabels([])
    axs[1][i].set_yticklabels([])
    axs[2][i].set_yticklabels([])

    axs[0][i].set_ylabel(None)
    axs[1][i].set_ylabel(None)
    axs[2][i].set_ylabel(None)


    # axs[0][i].hist(features[i])
    # axs[1][i].hist(features[i+6])
    # axs[2][i].hist(features[i+11])

plt.show()

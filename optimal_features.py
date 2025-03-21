import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



plt.style.use("default")
sns.set_style("ticks")





encoding_dim = 10
run = 1
epochs = 750
batch_size = 32


# dataframe to store the losses
df_loss = pd.DataFrame(columns=["Extracted Features", "Min Total", "Med Total", "Max Total", "Min Reconstruction", "Med Reconstruction", "Max Reconstruction", "Min KL", "Med KL", "Max KL"])

# load total losses for each run
total_loss_1 = np.load("Variational Eagle/Loss/Fully Balanced/total_loss_1.npy")
total_loss_2 = np.load("Variational Eagle/Loss/Fully Balanced/total_loss_2.npy")
total_loss_3 = np.load("Variational Eagle/Loss/Fully Balanced/total_loss_3.npy")

# load reconstruction losses for each run
reconstruction_loss_1 = np.load("Variational Eagle/Loss/Fully Balanced/reconstruction_loss_1.npy")
reconstruction_loss_2 = np.load("Variational Eagle/Loss/Fully Balanced/reconstruction_loss_2.npy")
reconstruction_loss_3 = np.load("Variational Eagle/Loss/Fully Balanced/reconstruction_loss_3.npy")

print(reconstruction_loss_1[25])
print(reconstruction_loss_2[25])
print(reconstruction_loss_3[25])
print()
print(reconstruction_loss_1[23])
print(reconstruction_loss_2[23])
print(reconstruction_loss_3[23])

# load kl losses for each run
kl_loss_1 = np.load("Variational Eagle/Loss/Fully Balanced/kl_loss_1.npy")
kl_loss_2 = np.load("Variational Eagle/Loss/Fully Balanced/kl_loss_2.npy")
kl_loss_3 = np.load("Variational Eagle/Loss/Fully Balanced/kl_loss_3.npy")

for i in range(len(total_loss_1)):

    # sort the total, reconstruction and kl loss for each run of that number of latent features
    total_sorted = np.sort(np.array([total_loss_1[i], total_loss_2[i], total_loss_3[i]]))
    reconstruction_sorted = np.sort(np.array([reconstruction_loss_1[i], reconstruction_loss_2[i], reconstruction_loss_3[i]]))
    kl_sorted = np.sort(np.array([kl_loss_1[i], kl_loss_2[i], kl_loss_3[i]]))

    # add the sorted values to the loss dataframe
    df_loss.loc[len(df_loss)] = [i+1] + list(total_sorted) + list(reconstruction_sorted) + list(kl_sorted)

df_loss = df_loss.tail(23)

print(df_loss)

# find the size of the loss error bars for total loss
total_err_upper = np.array(df_loss["Max Total"] - df_loss["Med Total"])
total_err_lower = np.array(df_loss["Med Total"] - df_loss["Min Total"])

# find the size of the loss error bars for reconstruction loss
reconstruction_err_upper = np.array(df_loss["Max Reconstruction"] - df_loss["Med Reconstruction"])
reconstruction_err_lower = np.array(df_loss["Med Reconstruction"] - df_loss["Min Reconstruction"])

# find the size of the loss error bars for kl divergence
kl_err_upper = np.array(df_loss["Max KL"] - df_loss["Med KL"])
kl_err_lower = np.array(df_loss["Med KL"] - df_loss["Min KL"])





# fig, axs = plt.subplots(3, 1, figsize=(12, 15))
#
# axs[0].errorbar(df_loss["Extracted Features"], df_loss["Med Total"], yerr=[total_err_lower, total_err_upper], fmt="o", label="750 Epoch, 32 Batch Size")
# axs[1].errorbar(df_loss["Extracted Features"], df_loss["Med Reconstruction"], yerr=[reconstruction_err_lower, reconstruction_err_upper], fmt="o", label="750 Epoch, 32 Batch Size")
# axs[2].errorbar(df_loss["Extracted Features"], df_loss["Med KL"], yerr=[kl_err_lower, kl_err_upper], fmt="o", label="750 Epoch, 32 Batch Size")
#
# # axs[0].set_xticks(range(2, 30, 2))
# # axs[1].set_xticks(range(2, 30, 2))
# # axs[2].set_xticks(range(2, 30, 2))
#
# axs[0].set_xticks(range(14, 29))
# axs[1].set_xticks(range(14, 29))
# axs[2].set_xticks(range(14, 29))
#
# axs[0].set_title("Total Loss")
# axs[1].set_title("Reconstruction Loss")
# axs[2].set_title("KL Loss")
#
# axs[0].set_ylabel("Loss")
# axs[1].set_ylabel("BCE Loss")
# axs[2].set_ylabel("KL Divergence")
#
# axs[0].set_xlabel("Latent Features")
# axs[1].set_xlabel("Latent Features")
# axs[2].set_xlabel("Latent Features")
#
#
# plt.savefig("Variational Eagle/Plots/loss_pot_zoomed", bbox_inches='tight')
# plt.show()




fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.errorbar(df_loss["Extracted Features"], df_loss["Med Reconstruction"], yerr=[reconstruction_err_lower, reconstruction_err_upper], fmt="o", label="750 Epoch, 32 Batch Size")

axs.set_xticks(range(6, 30, 2))

axs.set_title("Reconstruction Loss")
axs.set_ylabel("BCE Loss")
axs.set_xlabel("Latent Features")



x_val = np.array(list(df_loss["Extracted Features"]) + list(df_loss["Extracted Features"]) + list(df_loss["Extracted Features"])).reshape(-1, 1)
y_val = np.array(list(df_loss["Med Reconstruction"]) + list(df_loss["Min Reconstruction"]) + list(df_loss["Max Reconstruction"])).reshape(-1, 1)

fit = LinearRegression().fit(x_val, y_val)
fit_line = fit.predict(x_val)
axs.plot(x_val, fit_line, color="black")

print(fit.coef_, fit.intercept_)



plt.savefig("Variational Eagle/Plots/loss_pot_zoomed_reconstruction", bbox_inches='tight')
plt.show()




# loss of final batch

# df_loss = pd.DataFrame(columns=["Extracted Features", "Min Total", "Med Total", "Max Total", "Min Reconstruction", "Med Reconstruction", "Max Reconstruction", "Min KL", "Med KL", "Max KL"])
#
# for i in range(1, 21):
#     try:
#
#         # load the three different runs
#         loss_1 = list(np.load("Variational Eagle/Loss/Fully Balanced/" + str(i) + "_feature_750_epoch_32_bs_loss_1.npy"))
#         loss_2 = list(np.load("Variational Eagle/Loss/Fully Balanced/" + str(i) + "_feature_750_epoch_32_bs_loss_2.npy"))
#         loss_3 = list(np.load("Variational Eagle/Loss/Fully Balanced/" + str(i) + "_feature_750_epoch_32_bs_loss_3.npy"))
#
#         # sort the reconstruction loss and kl divergence
#         total_sorted = np.sort(np.array([loss_1[0], loss_2[0], loss_3[0]]))
#         reconstruction_sorted = np.sort(np.array([loss_1[1], loss_2[1], loss_3[1]]))
#         kl_sorted = np.sort(np.array([loss_1[2], loss_2[2], loss_3[2]]))
#
#         # dataframe to store order of losses (reconstruction and kl divergence)
#         df_loss.loc[len(df_loss)] = [i] + list(total_sorted) + list(reconstruction_sorted) + list(kl_sorted)
#
#     # if we don't have a run for this number of features, skip it
#     except:
#         print(i)
#
# print(df_loss)
#
#
# # find the size of the loss error bars for total loss
# total_err_upper = np.array(df_loss["Max Total"] - df_loss["Med Total"])
# total_err_lower = np.array(df_loss["Med Total"] - df_loss["Min Total"])
#
# # find the size of the loss error bars for reconstruction loss
# reconstruction_err_upper = np.array(df_loss["Max Reconstruction"] - df_loss["Med Reconstruction"])
# reconstruction_err_lower = np.array(df_loss["Med Reconstruction"] - df_loss["Min Reconstruction"])
#
# # find the size of the loss error bars for kl divergence
# kl_err_upper = np.array(df_loss["Max KL"] - df_loss["Med KL"])
# kl_err_lower = np.array(df_loss["Med KL"] - df_loss["Min KL"])
#
#
#
# fig, axs = plt.subplots(1, 1, figsize=(8, 4))
#
# # axs.errorbar(df_loss["Extracted Features"], df_loss["Med Total"], yerr=[total_err_lower, total_err_upper], fmt="o", label="750 Epoch, 32 Batch Size")
# axs.errorbar(df_loss["Extracted Features"], df_loss["Med Reconstruction"], yerr=[reconstruction_err_lower, reconstruction_err_upper], fmt="o", label="750 Epoch, 32 Batch Size")
# # axs.errorbar(df_loss["Extracted Features"], df_loss["Med KL"], yerr=[kl_err_lower, kl_err_upper], fmt="o", label="750 Epoch, 32 Batch Size")
#
# axs.set_ylabel("Loss")
# axs.set_xlabel("Extracted Features")








# Meaningful extracted features

fig, axs = plt.subplots(1, 1, figsize=(10, 5))

df_num = pd.DataFrame(columns=["Extracted Features", "Min", "Med", "Max"])
# df_num = pd.DataFrame(columns=["Extracted Features", "1, "2", "3"])

for i in range(1, 29):

    features_1 = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(i) + "_feature_750_epoch_32_bs_features_1.npy")[0]
    features_2 = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(i) + "_feature_750_epoch_32_bs_features_2.npy")[0]
    features_3 = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(i) + "_feature_750_epoch_32_bs_features_3.npy")[0]

    pca_1 = PCA(n_components=0.99).fit(features_1)
    pca_2 = PCA(n_components=0.99).fit(features_2)
    pca_3 = PCA(n_components=0.99).fit(features_3)

    num_1 = pca_1.components_.shape[0]
    num_2 = pca_2.components_.shape[0]
    num_3 = pca_3.components_.shape[0]

    sorted = np.sort(np.array([num_1, num_2, num_3]))

    df_num.loc[len(df_num)] = [i, sorted[0], sorted[1], sorted[2]]
    # df_num.loc[len(df_num)] = [i, i, num_2, num_3]

# find the size of the loss error bars for reconstruction loss
num_err_upper = np.array(df_num["Max"] - df_num["Med"])
num_err_lower = np.array(df_num["Med"] - df_num["Min"])

axs.errorbar(df_num["Extracted Features"], df_num["Med"], yerr=[num_err_lower, num_err_upper], fmt="o")

axs.set_xticks(range(2, 30, 2))
axs.set_xlabel("Latent Features")
axs.set_ylabel("Meaningful Extracted Features")

# plt.scatter(df_num["Extracted Features"], df_num["1"])
# plt.scatter(df_num["Extracted Features"], df_num["2"])
# plt.scatter(df_num["Extracted Features"], df_num["3"])

plt.show()













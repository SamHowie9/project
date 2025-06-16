import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random





run = 2
encoding_dim = 49
n_flows = 1
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32






# original model with no flows

# # dataframe to store the losses
# df_loss = pd.DataFrame(columns=["Extracted Features", "Min Total", "Med Total", "Max Total", "Min Reconstruction", "Med Reconstruction", "Max Reconstruction", "Min KL", "Med KL", "Max KL"])
#
# # load total losses for each run
# total_loss_1 = np.load("Variational Eagle/Loss/Final/total_loss_1.npy")
# total_loss_2 = np.load("Variational Eagle/Loss/Final/total_loss_2.npy")
# total_loss_3 = np.load("Variational Eagle/Loss/Final/total_loss_3.npy")
#
# # load reconstruction losses for each run
# reconstruction_loss_1 = np.load("Variational Eagle/Loss/Final/reconstruction_loss_1.npy")
# reconstruction_loss_2 = np.load("Variational Eagle/Loss/Final/reconstruction_loss_2.npy")
# reconstruction_loss_3 = np.load("Variational Eagle/Loss/Final/reconstruction_loss_3.npy")
#
# # print(reconstruction_loss_1[25])
# # print(reconstruction_loss_2[25])
# # print(reconstruction_loss_3[25])
# # print()
# # print(reconstruction_loss_1[23])
# # print(reconstruction_loss_2[23])
# # print(reconstruction_loss_3[23])
#
# # load kl losses for each run
# kl_loss_1 = np.load("Variational Eagle/Loss/Final/kl_loss_1.npy")
# kl_loss_2 = np.load("Variational Eagle/Loss/Final/kl_loss_2.npy")
# kl_loss_3 = np.load("Variational Eagle/Loss/Final/kl_loss_3.npy")
#
# total_loss_1 = reconstruction_loss_1 + (0.0001 * kl_loss_1)
# total_loss_2 = reconstruction_loss_2 + (0.0001 * kl_loss_2)
# total_loss_3 = reconstruction_loss_3 + (0.0001 * kl_loss_3)
#
#
#
# for i in range(len(total_loss_1)):
#
#     # sort the total, reconstruction and kl loss for each run of that number of latent features
#     total_sorted = np.sort(np.array([total_loss_1[i], total_loss_2[i], total_loss_3[i]]))
#     reconstruction_sorted = np.sort(np.array([reconstruction_loss_1[i], reconstruction_loss_2[i], reconstruction_loss_3[i]]))
#     kl_sorted = np.sort(np.array([kl_loss_1[i], kl_loss_2[i], kl_loss_3[i]]))
#
#     # add the sorted values to the loss dataframe
#     df_loss.loc[len(df_loss)] = [i+1] + list(total_sorted) + list(reconstruction_sorted) + list(kl_sorted)
#
# # df_loss = df_loss.tail(36)
#
# print(df_loss)
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









fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# axs[0].errorbar(df_loss["Extracted Features"], df_loss["Med Total"], yerr=[total_err_lower, total_err_upper], fmt="o", label="Original VAE")
# axs[1].errorbar(df_loss["Extracted Features"], df_loss["Med Reconstruction"], yerr=[reconstruction_err_lower, reconstruction_err_upper], fmt="o", label="Original VAE")
# axs[2].errorbar(df_loss["Extracted Features"], df_loss["Med KL"], yerr=[kl_err_lower, kl_err_upper], fmt="o", label="Original VAE")

# axs[0].set_xticks(range(15, 51, 5))
# axs[1].set_xticks(range(15, 51, 5))
# axs[2].set_xticks(range(15, 51, 5))

axs[0].set_xticks([1] + list(range(5, 51, 5)))
axs[1].set_xticks([1] + list(range(5, 51, 5)))
axs[2].set_xticks([1] + list(range(5, 51, 5)))

# axs[0].set_xticks(range(14, 29))
# axs[1].set_xticks(range(14, 29))
# axs[2].set_xticks(range(14, 29))

axs[0].set_title("Total Loss")
axs[1].set_title("Reconstruction Loss")
axs[2].set_title("KL Loss")

axs[0].set_ylabel("Loss")
axs[1].set_ylabel("BCE Loss")
axs[2].set_ylabel("KL Divergence")

axs[0].set_xlabel("Latent Features")
axs[1].set_xlabel("Latent Features")
axs[2].set_xlabel("Latent Features")










for n_flows in [0]:

    for run in [1, 2, 3]:

        total_loss_all = []
        reconstruction_loss_all = []
        kl_loss_all = []

        loss_all = pd.DataFrame(columns=["feature", "total_loss", "reconstruction_loss", "kl_loss"])

        for encoding_dim in range(10, 51):
        # for encoding_dim in [30]:

            try:
                losses = np.load("Variational Eagle/Loss/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
                loss_all.loc[len(loss_all)] = [encoding_dim] + list(losses)
            except:
                print(encoding_dim, n_flows, run)

        # axs[0].plot(loss_all["feature"], loss_all["total_loss"], label="No Flows (New)", c="black")
        # axs[1].plot(loss_all["feature"], loss_all["reconstruction_loss"], label="No Flows (New)", c="black")
        # axs[2].plot(loss_all["feature"], loss_all["kl_loss"], label="No Flows (New)", c="black")

        axs[0].scatter(loss_all["feature"], loss_all["total_loss"], label="Run "+str(run)+", Flows "+str(n_flows))
        axs[1].scatter(loss_all["feature"], loss_all["reconstruction_loss"], label="Run "+str(run)+", Flows "+str(n_flows))
        axs[2].scatter(loss_all["feature"], loss_all["kl_loss"], label="Run "+str(run)+", Flows "+str(n_flows))

        print(loss_all)

axs[0].legend()
axs[1].legend()
axs[2].legend()





# total_loss_flow_1 = []
# total_loss_flow_2 = []
# total_loss_flow_3 = []
#
# reconstruction_loss_flow_1 = []
# reconstruction_loss_flow_2 = []
# reconstruction_loss_flow_3 = []
#
# kl_loss_flow_1 = []
# kl_loss_flow_2 = []
# kl_loss_flow_3 = []
#
# flow_range = range(5, 49)
# run=2
#
# for i in flow_range:
#
#     # try:
#     total_loss = np.load("Variational Eagle/Loss/Normalising Flow/total_loss_beta_" + str(i) + "_" + str(run) + ".npy")
#     reconstruction_loss = np.load("Variational Eagle/Loss/Normalising Flow/reconstruction_loss_beta_" + str(i) + "_" + str(run) + ".npy")
#     kl_loss = np.load("Variational Eagle/Loss/Normalising Flow/kl_loss_beta_" + str(i) + "_" + str(run) + ".npy")
#
#     total_loss_flow_1.append(total_loss[0])
#     total_loss_flow_2.append(total_loss[1])
#     total_loss_flow_3.append(total_loss[2])
#
#     reconstruction_loss_flow_1.append(reconstruction_loss[0])
#     reconstruction_loss_flow_2.append(reconstruction_loss[1])
#     reconstruction_loss_flow_3.append(reconstruction_loss[2])
#
#     kl_loss_flow_1.append(kl_loss[0])
#     kl_loss_flow_2.append(kl_loss[1])
#     kl_loss_flow_3.append(kl_loss[2])
#
#     # except:
#     #     print(i)
#
#
# print(len(total_loss_flow_1))
# print(len(total_loss_flow_2))
# print(len(total_loss_flow_3))
# print()
# print(len(reconstruction_loss_flow_1))
# print(len(reconstruction_loss_flow_2))
# print(len(reconstruction_loss_flow_3))
# print()
# print(len(kl_loss_flow_1))
# print(len(kl_loss_flow_2))
# print(len(kl_loss_flow_3))
#
#
# # axs[0].plot(flow_range, total_loss_flow_1, label="1 Flow Layer", c="C9")
# # axs[0].plot(flow_range, total_loss_flow_2, label="2 Flow Layers", c="C1")
# axs[0].plot(flow_range, total_loss_flow_3, label="3 Flow Layers", c="C1")
#
# # axs[1].plot(flow_range, reconstruction_loss_flow_1, label="1 Flow Layer", c="C9")
# # axs[1].plot(flow_range, reconstruction_loss_flow_2, label="2 Flow Layers", c="C1")
# axs[1].plot(flow_range, reconstruction_loss_flow_3, label="3 Flow Layers", c="C1")
# axs[1].ticklabel_format(style='plain', axis='both')
# axs[1].get_yaxis().get_major_formatter().set_useOffset(False)
#
# # axs[2].plot(flow_range, kl_loss_flow_1, label="1 Flow Layer", c="C9")
# # axs[2].plot(flow_range, kl_loss_flow_2, label="2 Flow Layers", c="C1")
# axs[2].plot(flow_range, kl_loss_flow_3, label="3 Flow Layers", c="C1")
#
# axs[0].legend()
# axs[1].legend()
# axs[2].legend()

# axs[0].set_xlim(9.5, 50.5)
# axs[1].set_xlim(9.5, 50.5)
# axs[2].set_xlim(9.5, 50.5)
#
# axs[0].set_ylim(0.209, 0.2125)
# axs[1].set_ylim(0.209, 0.2125)
# axs[2].set_ylim(0.55, 2.5)

plt.savefig("Variational Eagle/Plots/feature_loss", bbox_inches='tight')
plt.show()



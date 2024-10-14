import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# # dataframe containing all losses
# # df_loss = pd.DataFrame(columns=["Extracted Features", "Loss", "Reconstruction Loss", "KL Loss"])
# df_loss = pd.DataFrame(columns=["Extracted Features", "Min Loss", "Min KL", "Med Loss", "Med KL", "Max Loss", "Max KL"])
#
# for i in range(4, 51):
#     try:
#
#         loss_1 = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_1.npy"))
#         loss_2 = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_2.npy"))
#         loss_3 = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_3.npy"))
#
#         loss_sorted = np.sort(np.array([loss_1[1], loss_2[1], loss_3[1]]))
#         kl_sorted = np.sort(np.array([loss_1[2], loss_2[2], loss_3[2]]))
#
#         # print([loss_1[1], loss_2[1], loss_3[1]])
#         # print(np.sort(np.array([loss_1[1], loss_2[1], loss_3[1]])))
#         # print(([loss_1[1], loss_2[1], loss_3[1]].sort())[1])
#
#         # min_loss = loss_sorted[0]
#         # med_loss = loss_sorted[1]
#         # max_loss = loss_sorted[2]
#         #
#         # min_kl = kl_sorted[0]
#         # med_kl = kl_sorted[1]
#         # max_kl = kl_sorted[2]
#
#         df_loss.loc[len(df_loss)] = [i, loss_sorted[0], kl_sorted[0], loss_sorted[1], kl_sorted[1], loss_sorted[2], kl_sorted[2]]
#
#
#
#         # loss = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_3.npy"))
#         # df_loss.loc[len(df_loss)] = [i] + loss
#
#
#         print("   " + str(i))
#     except:
#         print(i)
#
# print(df_loss)
#
# loss_err_upper = np.array(df_loss["Max Loss"] - df_loss["Med Loss"])
# loss_err_lower = np.array(df_loss["Med Loss"] - df_loss["Min Loss"])
#
# kl_err_upper = np.array(df_loss["Max KL"] - df_loss["Med KL"])
# kl_err_lower = np.array(df_loss["Med Loss"] - df_loss["Min Loss"])
#
# print(loss_err_upper)
#
# fig, axs1 = plt.subplots()
#
# # axs1.plot(df_loss["Extracted Features"], df_loss["Med Loss"])
# # axs1.plot(df_loss["Extracted Features"], df_loss["Min Loss"])
# # axs1.plot(df_loss["Extracted Features"], df_loss["Max Loss"])
# axs1.errorbar(df_loss["Extracted Features"], df_loss["Med Loss"], yerr=[loss_err_lower, loss_err_upper], fmt="o")
# axs1.set_ylabel("Loss")
# axs1.set_xlabel("Extracted Features")
#
# axs2 = axs1.twinx()
#
# # axs2.plot(df_loss["Extracted Features"], df_loss["Med KL"], color="red")
# axs2.errorbar(df_loss["Extracted Features"], df_loss["Med KL"], yerr=[kl_err_lower, kl_err_upper], fmt="o", color="red")
# axs2.set_ylabel("KL Loss")
#
# plt.show()






df_vae_train = pd.DataFrame(columns=["Extracted Features", "RMSE 1", "RMSE 2", "RMSE 3"])
df_vae_test = pd.DataFrame(columns=["Extracted Features", "RMSE 1", "RMSE 2", "RMSE 3"])

df_cae_train = pd.DataFrame(columns=["Extracted Features", "RMSE 1", "RMSE 2", "RMSE 3"])
df_cae_test = pd.DataFrame(columns=["Extracted Features", "RMSE 1", "RMSE 2", "RMSE 3"])

df_vae_train["Extracted Features"] = df_vae_test["Extracted Features"] = df_cae_train["Extracted Features"] = df_cae_test["Extracted Features"] = range(1, 51)

df_vae_train["RMSE 1"] = np.load("Variational Eagle/Loss/rmse_train_1.npy")
df_vae_train["RMSE 2"] = np.load("Variational Eagle/Loss/rmse_train_2.npy")
df_vae_train["RMSE 3"] = np.load("Variational Eagle/Loss/rmse_train_3.npy")

df_vae_test["RMSE 1"] = np.load("Variational Eagle/Loss/rmse_test_1.npy")
df_vae_test["RMSE 2"] = np.load("Variational Eagle/Loss/rmse_test_2.npy")
df_vae_test["RMSE 3"] = np.load("Variational Eagle/Loss/rmse_test_3.npy")

for i in range(1, 51):
    df_cae_train.loc[i-1, "RMSE 1"] = np.load("Convolutional Eagle/Loss Rand/" + str(i) + "_feature_loss_1.npy")[-1]
    df_cae_train.loc[i-1, "RMSE 2"] = np.load("Convolutional Eagle/Loss Rand/" + str(i) + "_feature_loss_2.npy")[-1]
    df_cae_train.loc[i-1, "RMSE 3"] = np.load("Convolutional Eagle/Loss Rand/" + str(i) + "_feature_loss_3.npy")[-1]

# for i in range(1, 51):
#     df_cae_test.loc[i-1, "RMSE 1"] = np.load("Convolutional Eagle/Loss Rand/" + str(i) + "_feature_loss_1.npy")[-1]
#     df_cae_test.loc[i-1, "RMSE 2"] = np.load("Convolutional Eagle/Loss Rand/" + str(i) + "_feature_loss_2.npy")[-1]
#     df_cae_test.loc[i-1, "RMSE 3"] = np.load("Convolutional Eagle/Loss Rand/" + str(i) + "_feature_loss_3.npy")[-1]

print(df_vae_test)

plt.plot(df_vae_train["Extracted Features"], df_vae_train["RMSE 1"], label="Variational Train")
plt.plot(df_vae_test["Extracted Features"], df_vae_test["RMSE 1"], label="Variational Test")
plt.plot(df_cae_train["Extracted Features"], df_cae_train["RMSE 2"], label="Convolutional Train")
plt.legend()
plt.show()



# df_vae_train.assign(RMSE_1 = np.load("Variational Eagle/Loss/rmse_train_1.npy"), RMSE_2 = np.load("Variational Eagle/Loss/rmse_train_2.npy"), RMSE_3 = np.load("Variational Eagle/Loss/rmse_train_3.npy"))


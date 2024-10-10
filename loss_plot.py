import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# dataframe containing all losses
# df_loss = pd.DataFrame(columns=["Extracted Features", "Loss", "Reconstruction Loss", "KL Loss"])
df_loss = pd.DataFrame(columns=["Extracted Features", "Min Loss", "Min KL", "Med Loss", "Med KL", "Max Loss", "Max KL"])

for i in range(1, 51):
    try:

        loss_1 = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_1.npy"))
        loss_2 = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_2.npy"))
        loss_3 = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_3.npy"))

        loss_sorted = np.sort(np.array([loss_1[1], loss_2[1], loss_3[1]]))
        kl_sorted = np.sort(np.array([loss_1[2], loss_2[2], loss_3[2]]))

        # print([loss_1[1], loss_2[1], loss_3[1]])
        # print(np.sort(np.array([loss_1[1], loss_2[1], loss_3[1]])))
        # print(([loss_1[1], loss_2[1], loss_3[1]].sort())[1])

        # min_loss = loss_sorted[0]
        # med_loss = loss_sorted[1]
        # max_loss = loss_sorted[2]
        #
        # min_kl = kl_sorted[0]
        # med_kl = kl_sorted[1]
        # max_kl = kl_sorted[2]

        df_loss.loc[len(df_loss)] = [i, loss_sorted[0], kl_sorted[0], loss_sorted[1], kl_sorted[1], loss_sorted[2], kl_sorted[2]]



        # loss = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_3.npy"))
        # df_loss.loc[len(df_loss)] = [i] + loss


        print("   " + str(i))
    except:
        print(i)

print(df_loss)

loss_err_upper = np.array(df_loss["Max Loss"] - df_loss["Med Loss"])
loss_err_lower = np.array(df_loss["Med Loss"] - df_loss["Min Loss"])

kl_err_upper = np.array(df_loss["Max KL"] - df_loss["Med KL"])
kl_err_lower = np.array(df_loss["Med Loss"] - df_loss["Min Loss"])

print(loss_err_upper)

fig, axs1 = plt.subplots()

# axs1.plot(df_loss["Extracted Features"], df_loss["Med Loss"])
# axs1.plot(df_loss["Extracted Features"], df_loss["Min Loss"])
# axs1.plot(df_loss["Extracted Features"], df_loss["Max Loss"])
axs1.errorbar(df_loss["Extracted Features"], df_loss["Med Loss"], yerr=[loss_err_lower, loss_err_upper], fmt="o")
axs1.set_ylabel("Loss")
axs1.set_xlabel("Extracted Features")

axs2 = axs1.twinx()

# axs2.plot(df_loss["Extracted Features"], df_loss["Med KL"], color="red")
axs2.errorbar(df_loss["Extracted Features"], df_loss["Med KL"], yerr=[kl_err_lower, kl_err_upper], fmt="o", color="red")
axs2.set_ylabel("KL Loss")

plt.show()

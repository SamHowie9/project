import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# dataframe containing all losses
df_loss = pd.DataFrame(columns=["Extracted Features", "Loss", "Reconstruction Loss", "KL Loss"])

for i in range(4, 50):
    try:
        loss = list(np.load("Variational Eagle/Loss/" + str(i) + "_feature_300_epoch_loss_1.npy"))
        df_loss.loc[len(df_loss)] = [i] + loss
        print("   " + str(i))
    except:
        print(i)


fig, axs1 = plt.subplots()

axs1.plot(df_loss["Extracted Features"], df_loss["Loss"])
axs1.set_ylabel("Loss")
axs1.set_xlabel("Extracted Features")

axs2 = axs1.twinx()

axs2.plot(df_loss["Extracted Features"], df_loss["KL Loss"], color="red")
axs2.set_ylabel("KL Loss")

plt.show()

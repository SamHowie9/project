from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



# load pca and vae features
pca_features = np.load("Variational Eagle/Extracted Features/PCA/pca_features_15_features.npy")
vae_features = np.load("Variational Eagle/Extracted Features/Normalised Individually/20_feature_300_epoch_features_3.npy")[0]

# apply pca on vae features to get vae + pca features
pca = PCA(n_components=11).fit(vae_features)
vae_pca_features = pca.transform(vae_features)





# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# account for the validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")



# find the bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
print(bad_fit)

# remove these galaxies
for i, galaxy in enumerate(bad_fit):
    pca_features = np.delete(pca_features, galaxy-i, 0)
    vae_features = np.delete(vae_features, galaxy - i, 0)
    vae_pca_features = np.delete(vae_pca_features, galaxy - i, 0)
    all_properties = all_properties.drop(galaxy, axis=0)



print(all_properties)



fig, axs = plt.subplots(4, 3, figsize=(15, 20))


axs[0, 0].scatter(pca_features.T[3], all_properties["n_r"], s=0.5)
axs[0, 0].set_title("PCA", fontsize=15)
axs[0, 0].set_ylabel("Sersic Index", fontsize=15)

axs[0, 1].scatter(vae_features.T[4], all_properties["n_r"], s=0.5)
axs[0, 1].set_title("VAE (20 Extracted Features)", fontsize=15)

axs[0, 2].scatter(vae_pca_features.T[1], all_properties["n_r"], s=0.5)
axs[0, 2].set_title("VAE + PCA", fontsize=15)



axs[1, 0].scatter(pca_features.T[0], all_properties["re_r"], s=0.5)
axs[1, 0].set_ylabel("Semi-Major Axis", fontsize=15)

axs[1, 1].scatter(vae_features.T[13], all_properties["re_r"], s=0.5)

axs[1, 2].scatter(vae_pca_features.T[0], all_properties["re_r"], s=0.5)



axs[2, 0].scatter(pca_features.T[1], all_properties["pa_r"], s=0.5)
axs[2, 0].set_ylabel("Position Angle", fontsize=15)

axs[2, 1].scatter(vae_features.T[5], all_properties["pa_r"], s=0.5)

axs[2, 2].scatter(vae_pca_features.T[3], all_properties["pa_r"], s=0.5)



axs[3, 0].scatter(pca_features.T[1], all_properties["q_r"], s=0.5)
axs[3, 0].set_ylabel("Axis Ratio", fontsize=15)

axs[3, 1].scatter(vae_features.T[5], all_properties["q_r"], s=0.5)

axs[3, 2].scatter(vae_pca_features.T[3], all_properties["q_r"], s=0.5)



plt.savefig("Variational Eagle/Plots/pca_vs_vae", bbox_inches='tight')
plt.show()

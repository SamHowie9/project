from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


encoding_dim = 20


features_individual = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]
features_g = np.load("Variational Eagle/Extracted Features/Normalised to g/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]
features_r = np.load("Variational Eagle/Extracted Features/Normalised to r/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]





# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")



bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()

for i, galaxy in enumerate(bad_fit):
    features_individual = np.delete(features_individual, galaxy-i, 0)
    features_g = np.delete(features_g, galaxy-i, 0)
    features_r = np.delete(features_r, galaxy-i, 0)
    all_properties = all_properties.drop(galaxy, axis=0)



print(all_properties)


print(list(all_properties["n_r"]))
print(list(features_g.T[0]))



fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].scatter(features_individual.T[8], all_properties["n_r"], s=0.5)
axs[0].set_title("Normalised Individually (ρ = 0.47)")

axs[1].scatter(features_r.T[17], all_properties["n_r"], s=0.5)
axs[1].set_title("Normalised to r (ρ = 0.64)")

axs[2].scatter(features_g.T[19], all_properties["n_r"], s=0.5)
axs[2].set_title("Normalised to g (ρ = 0.61)")

plt.savefig("Variational Eagle/Plots/normalisation_comparison_sersic", bbox_inches='tight')
plt.show()
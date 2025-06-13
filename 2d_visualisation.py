import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP
import seaborn as sns





run = 2
encoding_dim = 30
n_flows = 3
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32





# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# load the non parametric properties (restructure the dataframe to match the others)
non_parametric_properties = pd.read_hdf("Galaxy Properties/Eagle Properties/Ref100N1504.hdf5", key="galface/r")
non_parametric_properties = non_parametric_properties.reset_index()
non_parametric_properties = non_parametric_properties.sort_values(by="GalaxyID")

# load the disk-total ratios
disk_total = pd.read_csv("Galaxy Properties/Eagle Properties/disk_to_total.csv", comment="#")


# add the non parametric properties, and the disk-total to the other properties dataframe
all_properties = pd.merge(all_properties, non_parametric_properties, on="GalaxyID")
all_properties = pd.merge(all_properties, disk_total, on="GalaxyID")

# find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 1) |
#                           (all_properties["flag_r"] == 4) |
#                           (all_properties["flag_r"] == 5) |
#                           (all_properties["flag_r"] == 6))].index.tolist()
#
# # remove those galaxies
# for galaxy in bad_fit:
#     all_properties = all_properties.drop(galaxy, axis=0)
#
# # reset the index values
# all_properties = all_properties.reset_index(drop=True)
#
# # account for the training data
# all_properties = all_properties.iloc[:-200]








# extracted_features_all = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
extracted_features_all = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

extracted_features_real = extracted_features_all[:len(all_properties)]




print(extracted_features_all.shape)
print(extracted_features_real.shape)

pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features_all)
extracted_features_all = pca.transform(extracted_features_all)
extracted_features_real = pca.transform(extracted_features_real)


print(extracted_features_all.shape)
print(extracted_features_real.shape)




def classify_morphology(dt):

    if dt > 0.2:
        return "Spiral"
    elif dt < 0.1:
        return "Elliptical"
    else:
        return "Transitional"

# def classify_morphology(n):
#
#     if n <= 2.5:
#         return "Spiral"
#     elif n >= 4:
#         return "Elliptical"
#     else:
#         return "Transitional"

morphology = all_properties["DiscToTotal"].apply(classify_morphology).tolist()
# morphology = all_properties["n_r"].apply(classify_morphology).tolist()

print(morphology)


# for i in range(extracted_features_all.shape[0] - len(morphology)):
#     morphology.append("Augmented")


for i in range(2394):
    morphology.append("Elliptical")
for i in range(2552):
    morphology.append("Transitional")


print(len(morphology))





tsne = TSNE(n_components=2, random_state=0).fit_transform(extracted_features_all)

# umap = UMAP().fit_transform(extracted_features)

print(tsne.shape)

fig, axs = plt.subplots(1, 1, figsize=(10, 10))

palette = ["C0", "C1", "#D3D3D3"]

size_map = {"Spiral": 20, "Elliptical": 20, "Transitional": 10}
sizes = [size_map[m] for m in morphology]

alpha_map = {"Spiral": 1.0, "Transitional": 0.3, "Elliptical": 1.0}
alpha = [alpha_map[m] for m in morphology]

# sns.scatterplot(x=tsne.T[0], y=tsne.T[1], ax=axs, linewidth=0, s=20)
# sns.scatterplot(x=tsne.T[0], y=tsne.T[1], ax=axs, hue=morphology, palette="colorblind", linewidth=0, s=20)
# sns.scatterplot(x=tsne.T[0], y=tsne.T[1], ax=axs, hue=morphology, palette=palette, linewidth=0, s=20)
sns.scatterplot(x=tsne.T[0], y=tsne.T[1], ax=axs, hue=morphology, palette=palette, linewidth=0, size=sizes, sizes=(10, 20), legend="brief")

handles, labels = axs.get_legend_handles_labels()
filtered = [(h, l) for h, l in zip(handles, labels) if not l.isdigit()]
handles_filtered, labels_filtered = zip(*filtered)
axs.legend(handles_filtered, labels_filtered)


axs.set_xlim(-90, 90)
axs.set_ylim(-90, 90)

plt.savefig("Variational Eagle/Plots/tsne_nop_flow_pca_morphology_" + str(n_flows), bbox_inches="tight")
plt.show()

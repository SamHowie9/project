import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns


run = 18
encoding_dim = 35
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32



pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)




all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")




extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
print(extracted_features.shape)

pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
extracted_features = pca.transform(extracted_features)
print(extracted_features.shape)




# def classify_morphology(dt):
#
#     if dt > 0.2:
#         return "Spiral"
#     elif dt < 0.1:
#         return "Elliptical"
#     else:
#         return "Transitional"
#
# morphology = all_properties["DiscToTotal"].apply(classify_morphology).tolist()
# print(morphology)




# spirals_indices = all_properties[all_properties["DiscToTotal"] > 0.2].index.tolist()
# transitional_indices = all_properties[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")].index.tolist()
# elliptical_indices = all_properties[all_properties["DiscToTotal"] < 0.1].index.tolist()








fig, axs=plt.subplots(1, 1, figsize=(10, 10))
# plt.hist(all_properties[all_properties["re_r"] < 15]["re_r"])
plt.hist(all_properties["n_r"])
plt.show()









# umap = UMAP(n_components=2, init="spectral", random_state=0, n_neighbors=100).fit_transform(extracted_features)
# np.save("Variational Eagle/2D Visualisation/umap_pca.npy", umap)

umap = np.load("Variational Eagle/2D Visualisation/umap_spectral.npy")


# norm = TwoSlopeNorm(vmin=all_properties["n_r"].min(), vcenter=1.5, vmax=all_properties["n_r"].max())
norm = TwoSlopeNorm(vmin=all_properties["n_r"].min(), vcenter=1.5, vmax=5)
# norm = TwoSlopeNorm(vmin=all_properties["MassType_Star"].min(), vmax=0.25e12)


fig, axs = plt.subplots(1, 1, figsize=(14, 10))

# scatter = axs.scatter(x=umap.T[0], y=umap.T[1], s=10)
scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["n_r"], cmap="RdYlBu_r", norm=norm, s=10)
# scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["MassType_Star"], cmap="RdYlBu", norm=norm, s=10)
# scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["rhalf_ellip"], cmap="RdYlBu", vmin=all_properties["re_r"].min(), vmax=15, s=10)
# scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["rhalf_ellip"], cmap="RdYlBu", s=10)

cbar = plt.colorbar(scatter, ax=axs, label="Sersic Index")
cbar.ax.yaxis.set_label_position('left')


# scatter = axs.scatter(x=umap, y=all_properties["n_r"], s=10)
# axs.set_ylabel("Sersic Index")
# axs.set_xlabel("1D UMAP")



plt.savefig("Variational Eagle/2D Visualisation/umap_sersic_" + str(encoding_dim) + "_" + str(run) + "_15_4", bbox_inches="tight")
# plt.savefig("Variational Eagle/2D Visualisation/pca", bbox_inches="tight")
plt.show()

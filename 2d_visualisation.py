import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
import random


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

# extracted_features = extracted_features[:all_properties.shape[0]]


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








# fig, axs=plt.subplots(1, 1, figsize=(10, 10))
# # plt.hist(all_properties[all_properties["StarFormationRate"] < 1]["StarFormationRate"])
# # plt.hist(all_properties[all_properties["asymmetry"].between(2, 4.5)]["asymmetry"])
# plt.hist(all_properties["g-r"])
# # plt.hist(np.log10(all_properties["StarFormationRate"]))
# # sfr = np.log10(all_properties["StarFormationRate"].replace(0, all_properties["StarFormationRate"][all_properties["StarFormationRate"] != 0].min()))
# # plt.hist(sfr)
# plt.show()




print(all_properties[all_properties["StarFormationRate"] == 0])
print(all_properties[all_properties["StarFormationRate"] < 0])
print(all_properties.loc[all_properties["StarFormationRate"] != 0, "StarFormationRate"].min())




print(all_properties.shape)

n_neighbors=200
# umap = UMAP(n_components=2, init="spectral", random_state=0, n_neighbors=n_neighbors).fit_transform(extracted_features)
# np.save("Variational Eagle/2D Visualisation/umap_spectral_" + str(n_neighbors) + ".npy", umap)

umap = np.load("Variational Eagle/2D Visualisation/umap_spectral_" + str(n_neighbors) + ".npy")

# extracted_features = extracted_features[:all_properties.shape[0]]
# umap = umap[:all_properties.shape[0]]



# fig, axs = plt.subplots(1, 1, figsize=(14, 10))
#
# spiral_indices = all_properties.index[all_properties["DiscToTotal"] > 0.2].tolist()
#
# # norm = TwoSlopeNorm(vmin=all_properties["n_r"].min(), vcenter=1.5, vmax=all_properties["n_r"].max())
# # norm = TwoSlopeNorm(vmin=all_properties["DiscToTotal"].min(), vcenter=0.1, vmax=all_properties["DiscToTotal"].max())
# # norm = TwoSlopeNorm(vmin=all_properties["MassType_Star"].min(), vmax=0.25e12)
#
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], s=10)
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["n_r"], cmap="RdYlBu_r", norm=norm, s=10)
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["DiscToTotal"], cmap="RdYlBu", norm=norm, s=10)
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["MassType_Star"], cmap="RdYlBu_r", vmin=all_properties["MassType_Star"].min(), vmax=0.5e11, s=10)
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["re_r"], cmap="RdYlBu_r", vmin=all_properties["re_r"].min(), vmax=15, s=10)
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["rhalf_ellip"], cmap="RdYlBu_r", vmin=all_properties["rhalf_ellip"].min(), vmax=50, s=10)
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["g-i"], cmap="RdYlBu_r", vmin=all_properties["g-i"].min(), vmax=all_properties["g-i"].max(), s=10)
# # scatter = axs.scatter(x=umap.T[0], y=umap.T[1], c=all_properties["StarFormationRate"], vmin=all_properties["StarFormationRate"].min(), vmax=2, cmap="RdYlBu", s=10)
# scatter = axs.scatter(x=umap[spiral_indices].T[0], y=umap[spiral_indices].T[1], c=abs(all_properties[all_properties["DiscToTotal"] > 0.2]["pa_r"]), cmap="Greys", s=10)
#
# cbar = plt.colorbar(scatter, ax=axs, label="Position Angle")
# cbar.ax.yaxis.set_label_position('left')
# # cbar.set_ticks([0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8])
#
#
# # scatter = axs.scatter(x=umap, y=all_properties["n_r"], s=10)
# # axs.set_ylabel("Sersic Index")
# # axs.set_xlabel("1D UMAP")
#
#
# # rect_1 = patches.Rectangle((7.5, 3.1), 3.2, 2.2, linewidth=1, edgecolor='black', facecolor='none')
# # rect_2 = patches.Rectangle((5, -0.6), 1, 1.22, linewidth=1, edgecolor='black', facecolor='none')
# # rect_3 = patches.Rectangle((8.2, -0.5), 1.8, 1.5, linewidth=1, edgecolor='black', facecolor='none')
# #
# # axs.add_patch(rect_1)
# # axs.add_patch(rect_2)
# # axs.add_patch(rect_3)
#
#
# plt.savefig("Variational Eagle/2D Visualisation/umap_position_angle_" + str(encoding_dim) + "_" + str(run), bbox_inches="tight")
# # plt.savefig("Variational Eagle/2D Visualisation/pca", bbox_inches="tight")
# plt.show()









# all properties

n_norm = TwoSlopeNorm(vmin=all_properties["n_r"].min(), vcenter=1.5, vmax=6)
dt_norm = TwoSlopeNorm(vmin=all_properties["DiscToTotal"].min(), vcenter=0.1, vmax=all_properties["DiscToTotal"].max())

fig, axs = plt.subplots(3, 3, figsize=(30, 20))


dt_scatter = axs[0][0].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["DiscToTotal"], cmap="RdYlBu", norm=dt_norm, s=2)
axs[0][0].set_title("D/T", fontsize=20)
cbar = plt.colorbar(dt_scatter, ax=axs[0][0], label="D/T", pad=0.08)
cbar.set_label(label="D/T", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
# cbar.set_ticks([0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8])
# cbar.set_ticks(cbar.get_ticks()[::-1])
# cbar.set_ticklabels(cbar.get_ticks()[::-1])
cbar.set_ticks([0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02])
cbar.ax.tick_params(labelsize=20)
cbar.ax.invert_yaxis()

n_scatter = axs[0][1].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["n_r"], cmap="RdYlBu_r", norm=n_norm, s=2)
axs[0][1].set_title("Sèrsic Index", fontsize=20)
cbar = plt.colorbar(n_scatter, ax=axs[0][1], label="Sèrsic Index", pad=0.08)
cbar.set_label(label="Sèrsic Index", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
cbar.set_ticks([0.5, 1, 1.5, 2, 4, 6])
cbar.ax.tick_params(labelsize=20)

r_scatter = axs[0][2].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["rhalf_ellip"], cmap="RdYlBu_r", vmax=50, s=2)
axs[0][2].set_title("Half-light Radius", fontsize=20)
cbar = plt.colorbar(r_scatter, ax=axs[0][2], label="Half-light Radius (kpc)", pad=0.08)
cbar.set_label(label="Half-light Radius (kpc)", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
# cbar.set_ticks(cbar.get_ticks()[::-1])
# cbar.set_ticklabels(cbar.get_ticks()[::-1])
cbar.set_ticks([10, 20, 30, 40, 50])
cbar.ax.tick_params(labelsize=20)




c_scatter = axs[1][0].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["concentration"], vmin=2, vmax=4.5, cmap="RdYlBu_r", s=2)
axs[1][0].set_title("Concentration", fontsize=20)
cbar = plt.colorbar(c_scatter, ax=axs[1][0], label="Concentration", pad=0.08)
cbar.set_label(label="Concentration", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
cbar.ax.tick_params(labelsize=20)

a_scatter = axs[1][1].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["asymmetry"], vmax=0.3, cmap="RdYlBu", s=2)
axs[1][1].set_title("Asymmetry", fontsize=20)
cbar = plt.colorbar(a_scatter, ax=axs[1][1], label="Asymmetry", pad=0.08)
cbar.set_label(label="Asymmetry", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
# cbar.set_ticks(cbar.get_ticks()[::-1])
# cbar.set_ticklabels(cbar.get_ticks()[::-1])
cbar.set_ticks([0.3, 0.25, 0.2 , 0.15, 0.1, 0.05])
cbar.ax.invert_yaxis()
cbar.ax.tick_params(labelsize=20)

s_scatter = axs[1][2].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["smoothness"], vmax=0.05, cmap="RdYlBu", s=2)
axs[1][2].set_title("Smoothness", fontsize=20)
cbar = plt.colorbar(s_scatter, ax=axs[1][2], label="Smoothness", pad=0.08)
cbar.set_label(label="Smoothness", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
# cbar.set_ticks(cbar.get_ticks()[::-1])
# cbar.set_ticklabels(cbar.get_ticks()[::-1])
cbar.set_ticks([0.05, 0.04, 0.03, 0.02, 0.01, 0])
cbar.ax.tick_params(labelsize=20)
cbar.ax.invert_yaxis()




sfr = np.log10(all_properties["StarFormationRate"].replace(0, all_properties["StarFormationRate"][all_properties["StarFormationRate"] != 0].min()))
# sfr_scatter = axs[0].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["StarFormationRate"], cmap="RdYlBu_r", vmax=2, s=2)
sfr_scatter = axs[2][0].scatter(x=umap.T[0], y=umap.T[1], c=sfr, cmap="RdYlBu", vmin=-2, vmax=1, s=2)
axs[2][0].set_title("Star Formation Rate", fontsize=20)
# cbar = plt.colorbar(sfr_scatter, ax=axs[2][0], label="Log(SFR)", pad=0.08)
# cbar.set_label(label="Log(SFR)", fontsize=20)
cbar = plt.colorbar(sfr_scatter, ax=axs[2][0], label=r"$\log\left( \mathrm{SFR} \right)/M_{\odot}~\mathrm{yr}^{-1}$", pad=0.08)
cbar.set_label(label=r"$\log\left( \mathrm{SFR} \right)/M_{\odot}~\mathrm{yr}^{-1}$", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
cbar.ax.tick_params(labelsize=20)
cbar.ax.invert_yaxis()

gr_scatter = axs[2][1].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["g-r"], vmin=0.45, vmax=0.8, cmap="RdYlBu_r", s=2)
axs[2][1].set_title("g-r", fontsize=20)
cbar = plt.colorbar(gr_scatter, ax=axs[2][1], label="g-r", pad=0.08)
cbar.set_label(label="g-r", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
cbar.ax.tick_params(labelsize=20)

# sm_scatter = axs[2].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["MassType_Star"], vmax=0.5e11, cmap="RdYlBu_r", s=2)
sm_scatter = axs[2][2].scatter(x=umap.T[0], y=umap.T[1], c=np.log10(all_properties["MassType_Star"]), vmax=11.5, cmap="RdYlBu_r", s=2)
axs[2][2].set_title("Stellar Mass", fontsize=20)
# cbar = plt.colorbar(sm_scatter, ax=axs[2][2], label="Log(Stellar Mass)", pad=0.08)
# cbar.set_label(label="Log(Stellar Mass)", fontsize=20)
cbar = plt.colorbar(sm_scatter, ax=axs[2][2], label=r"$\log\left( M_* \right)/M_{\odot}$", pad=0.08)
cbar.set_label(label=r"$\log\left( M_* \right)/M_{\odot}$", fontsize=20)
cbar.ax.yaxis.set_label_position('left')
cbar.ax.tick_params(labelsize=20)



for ax_row in axs:
    for ax in ax_row:
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.tick_params(labelsize=20)



plt.savefig("Variational Eagle/2D Visualisation/umap_all_properties_" + str(n_neighbors) + "_balanced", bbox_inches="tight")
plt.savefig("Variational Eagle/2D Visualisation/umap_all_properties_" + str(n_neighbors) + "_balanced.pdf", bbox_inches="tight")
plt.show()












# all structure measurements

# n_norm = TwoSlopeNorm(vmin=all_properties["n_r"].min(), vcenter=1.5, vmax=all_properties["n_r"].max())
# dt_norm = TwoSlopeNorm(vmin=all_properties["DiscToTotal"].min(), vcenter=0.1, vmax=all_properties["DiscToTotal"].max())
# # sm_norm = TwoSlopeNorm(vmin=all_properties["MassType_Star"].min(), vmax=0.25e12)
#
#
# fig, axs = plt.subplots(2, 4, figsize=(35, 12))
#
#
# dt_scatter = axs[0][0].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["DiscToTotal"], cmap="RdYlBu_r", norm=dt_norm, s=2)
# axs[0][0].set_title("D/T", fontsize=20)
# cbar = plt.colorbar(dt_scatter, ax=axs[0][0], label="D/T", pad=0.08)
# cbar.set_label(label="D/T", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# # cbar.set_ticks([0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8])
# # cbar.set_ticks(cbar.get_ticks()[::-1])
# # cbar.set_ticklabels(cbar.get_ticks()[::-1])
# cbar.set_ticks([0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02])
# cbar.ax.tick_params(labelsize=20)
#
# n_scatter = axs[0][1].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["n_r"], cmap="RdYlBu_r", norm=n_norm, s=2)
# axs[0][1].set_title("Sèrsic Index", fontsize=20)
# cbar = plt.colorbar(n_scatter, ax=axs[0][1], label="Sèrsic Index", pad=0.08)
# cbar.set_label(label="Sèrsic Index", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# cbar.set_ticks([0.5, 1, 1.5, 2, 4, 6, 8])
# cbar.ax.tick_params(labelsize=20)
#
# q_scatter = axs[0][2].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["q_r"], cmap="RdYlBu_r", s=2)
# axs[0][2].set_title("Axis Ratio", fontsize=20)
# cbar = plt.colorbar(q_scatter, ax=axs[0][2], label="Axis Ratio", pad=0.08)
# cbar.set_label(label="Axis Ratio", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# cbar.ax.tick_params(labelsize=20)
#
# pa_scatter = axs[0][3].scatter(x=umap.T[0], y=umap.T[1], c=abs(all_properties["pa_r"]), cmap="RdYlBu_r", s=2)
# axs[0][3].set_title("Position Angle", fontsize=20)
# cbar = plt.colorbar(pa_scatter, ax=axs[0][3], label="Position Angle", pad=0.08)
# cbar.set_label(label="Position Angle", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# cbar.ax.tick_params(labelsize=20)
#
#
#
#
# r_scatter = axs[1][0].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["rhalf_ellip"], cmap="RdYlBu_r", vmax=50, s=2)
# axs[1][0].set_title("Half-light Radius (kpc)", fontsize=20)
# cbar = plt.colorbar(r_scatter, ax=axs[1][0], label="Half-Light Radius", pad=0.08)
# cbar.set_label(label="Half-light Radius", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# # cbar.set_ticks(cbar.get_ticks()[::-1])
# # cbar.set_ticklabels(cbar.get_ticks()[::-1])
# cbar.set_ticks([10, 20, 30, 40, 50])
# cbar.ax.tick_params(labelsize=20)
#
# c_scatter = axs[1][1].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["concentration"], vmin=2, vmax=4.5, cmap="RdYlBu_r", s=2)
# axs[1][1].set_title("Concentration", fontsize=20)
# cbar = plt.colorbar(c_scatter, ax=axs[1][1], label="Concentration", pad=0.08)
# cbar.set_label(label="Concentration", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# cbar.ax.tick_params(labelsize=20)
#
# a_scatter = axs[1][2].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["asymmetry"], vmax=0.3, cmap="RdYlBu", s=2)
# axs[1][2].set_title("Asymmetry", fontsize=20)
# cbar = plt.colorbar(a_scatter, ax=axs[1][2], label="Asymmetry", pad=0.08)
# cbar.set_label(label="Asymmetry", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# # cbar.set_ticks(cbar.get_ticks()[::-1])
# # cbar.set_ticklabels(cbar.get_ticks()[::-1])
# cbar.set_ticks([0.3, 0.25, 0.2 , 0.15, 0.1, 0.05])
# cbar.ax.invert_yaxis()
# cbar.ax.tick_params(labelsize=20)
#
# s_scatter = axs[1][3].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["smoothness"], vmax=0.05, cmap="RdYlBu_r", s=2)
# axs[1][3].set_title("Smoothness", fontsize=20)
# cbar = plt.colorbar(s_scatter, ax=axs[1][3], label="Smoothness", pad=0.08)
# cbar.set_label(label="Smoothness", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# # cbar.set_ticks(cbar.get_ticks()[::-1])
# # cbar.set_ticklabels(cbar.get_ticks()[::-1])
# cbar.set_ticks([0.05, 0.04, 0.03, 0.02, 0.01, 0])
# cbar.ax.tick_params(labelsize=20)
#
#
#
# for ax_row in axs:
#     for ax in ax_row:
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         ax.tick_params(labelsize=20)
#
#
# plt.savefig("Variational Eagle/2D Visualisation/umap_structure_measurements_" + str(encoding_dim) + "_" + str(run) + "_original", bbox_inches="tight")
# plt.show()
#
#
#
#
#
#
#
#
# # all physical properties
#
# n_norm = TwoSlopeNorm(vmin=all_properties["n_r"].min(), vcenter=1.5, vmax=all_properties["n_r"].max())
# dt_norm = TwoSlopeNorm(vmin=all_properties["DiscToTotal"].min(), vcenter=0.1, vmax=all_properties["DiscToTotal"].max())
# # sm_norm = TwoSlopeNorm(vmin=all_properties["MassType_Star"].min(), vmax=0.25e12)
#
#
# fig, axs = plt.subplots(1, 3, figsize=(25, 5))
#
#
# # sfr_scatter = axs[0].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["StarFormationRate"], cmap="RdYlBu_r", vmax=2, s=2)
# sfr_scatter = axs[0].scatter(x=umap.T[0], y=umap.T[1], c=np.log10(all_properties["StarFormationRate"]), cmap="RdYlBu_r", vmax=2, s=2)
# axs[0].set_title("Star Formation Rate", fontsize=20)
# cbar = plt.colorbar(sfr_scatter, ax=axs[0], label="Log(SFR)", pad=0.08)
# cbar.set_label(label="Log(SFR)", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# # cbar.set_ticks(cbar.get_ticks()[::-1])
# # cbar.set_ticklabels(cbar.get_ticks()[::-1])
# # cbar.set_ticks([2, 1.5, 1, 0.5, 0])
# cbar.set_ticks([0, 0.5, 1, 1.5, 2])
# cbar.ax.tick_params(labelsize=20)
#
# gr_scatter = axs[1].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["g-r"], cmap="RdYlBu_r", s=2)
# axs[1].set_title("g-r", fontsize=20)
# cbar = plt.colorbar(gr_scatter, ax=axs[1], label="g-r", pad=0.08)
# cbar.set_label(label="g-r", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# cbar.ax.tick_params(labelsize=20)
#
# # sm_scatter = axs[2].scatter(x=umap.T[0], y=umap.T[1], c=all_properties["MassType_Star"], vmax=0.5e11, cmap="RdYlBu_r", s=2)
# sm_scatter = axs[2].scatter(x=umap.T[0], y=umap.T[1], c=np.log10(all_properties["MassType_Star"]), vmax=11.5, cmap="RdYlBu_r", s=2)
# axs[2].set_title("Stellar Mass", fontsize=20)
# cbar = plt.colorbar(sm_scatter, ax=axs[2], label="Log(Stellar Mass)", pad=0.08)
# cbar.set_label(label="Log(Stellar Mass)", fontsize=20)
# cbar.ax.yaxis.set_label_position('left')
# cbar.ax.tick_params(labelsize=20)
#
#
#
# for ax in axs:
#     # ax.set_xticks([])
#     # ax.set_yticks([])
#     ax.tick_params(labelsize=20)
#
#
#
# plt.savefig("Variational Eagle/2D Visualisation/umap_physical_properties_" + str(encoding_dim) + "_" + str(run) + "_original", bbox_inches="tight")
# plt.show()







# random.seed(0)
#
# print(umap.shape)
#
# galaxies = []
#
# for i, (u1, u2) in enumerate(umap):
#     if (7.5 < u1 < 10.7) & (3.1 < u2 < 5.3):
#         galaxy = int(all_properties.iloc[i]["GalaxyID"])
#         galaxies.append(galaxy)
#
# print(random.sample(galaxies, 25))
#
#
#
# galaxies = []
#
# for i, (u1, u2) in enumerate(umap):
#     if (5 < u1 < 6) & (-0.6 < u2 < 0.62):
#         galaxy = int(all_properties.iloc[i]["GalaxyID"])
#         galaxies.append(galaxy)
#
# print(random.sample(galaxies, 25))
#
#
#
# galaxies = []
#
# for i, (u1, u2) in enumerate(umap):
#     if (8.2 < u1 < 10) & (-0.5 < u2 < 1):
#         galaxy = int(all_properties.iloc[i]["GalaxyID"])
#         galaxies.append(galaxy)
#
# print(random.sample(galaxies, 25))












random.seed(0)

print(umap.shape)


galaxies = []

for i, (u1, u2) in enumerate(umap):
    if (8.9 < u1 < 9.1) & (7.3 < u2 < 7.7):
        galaxy = int(all_properties.iloc[i]["GalaxyID"])
        galaxies.append(galaxy)

print(random.sample(galaxies, 9))



galaxies = []

for i, (u1, u2) in enumerate(umap):
    if (7.9 < u1 < 8.1) & (6.8 < u2 < 7.2):
        galaxy = int(all_properties.iloc[i]["GalaxyID"])
        galaxies.append(galaxy)

print(random.sample(galaxies, 9))



galaxies = []

for i, (u1, u2) in enumerate(umap):
    if (9.9 < u1 < 10.1) & (5.8 < u2 < 6.2):
        galaxy = int(all_properties.iloc[i]["GalaxyID"])
        galaxies.append(galaxy)

print(random.sample(galaxies, 9))



galaxies = []

for i, (u1, u2) in enumerate(umap):
    if (9.9 < u1 < 10.1) & (3.8 < u2 < 4.2):
        galaxy = int(all_properties.iloc[i]["GalaxyID"])
        galaxies.append(galaxy)

print(random.sample(galaxies, 9))



galaxies = []

for i, (u1, u2) in enumerate(umap):
    if (11.9 < u1 < 12.1) & (6.3 < u2 < 6.7):
        galaxy = int(all_properties.iloc[i]["GalaxyID"])
        galaxies.append(galaxy)

print(random.sample(galaxies, 9))



galaxies = []

for i, (u1, u2) in enumerate(umap):
    if (11.9 < u1 < 12.1) & (3.8 < u2 < 4.2):
        galaxy = int(all_properties.iloc[i]["GalaxyID"])
        galaxies.append(galaxy)

print(random.sample(galaxies, 9))
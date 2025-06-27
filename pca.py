from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random


run = 3
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 300
batch_size = 32




np.set_printoptions(linewidth=np.inf)








# pca scree plot

# fig, axs = plt.subplots(1, 2, figsize=(15, 6))
#
# for run in range(1, 26):
#
#     # extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_300_" + str(run) + ".npy")[0]
#     extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_750_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
#
#
#     print(len(extracted_features))
#
#     pca = PCA(n_components=encoding_dim, svd_solver="full").fit(extracted_features)
#     axs[0].plot(range(1, encoding_dim+1), pca.explained_variance_ratio_*100)
#     axs[1].plot(range(1, 16), pca.explained_variance_ratio_[:15]*100)
#     # print(pca.explained_variance_ratio_)
#     # print(pca.explained_variance_ratio_.shape)
#
#     pca = PCA(n_components=0.99, svd_solver="full").fit(extracted_features)
#     print(pca.explained_variance_ratio_.shape)
#
# axs[0].set_ylabel("Variance Explained (%)")
# axs[0].set_xlabel("Principal Components")
#
# axs[1].set_ylabel("Variance Explained (%)")
# axs[1].set_xlabel("Principal Components")
#
# # axs[0].set_xticks(range(1, encoding_dim+1))
# axs[1].set_xticks(range(1, 16))
#
# axs[0].legend()
# axs[1].legend()
#
# plt.savefig("Variational Eagle/Plots/pca_scree", bbox_inches="tight")
# plt.show()







# accumulation plot

# fig, axs = plt.subplots(1, 2, figsize=(15, 6))
#
# for run in range(1, 26):
#
#     # extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_300_" + str(run) + ".npy")[0]
#     extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_750_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
#
#     print(len(extracted_features))
#
#     pca = PCA(n_components=encoding_dim, svd_solver="full").fit(extracted_features)
#
#     accumulation = [pca.explained_variance_ratio_[0]*100]
#
#     print(accumulation)
#
#     for i in range(1, encoding_dim):
#         accumulation.append((accumulation[-1] + pca.explained_variance_ratio_[i]*100))
#         # print(accumulation)
#
#     axs[0].plot(range(1, encoding_dim + 1), accumulation)
#     axs[1].plot(range(1, 16), accumulation[:15])
#     # print(pca.explained_variance_ratio_)
#     # print(pca.explained_variance_ratio_.shape)
#
#     # pca = PCA(n_components=0.99, svd_solver="full").fit(extracted_features)
#     # print(pca.explained_variance_ratio_.shape)
#
#
# axs[0].set_ylabel("Variance Explained (%)")
# axs[0].set_xlabel("Principal Components")
#
# axs[1].set_ylabel("Variance Explained (%)")
# axs[1].set_xlabel("Principal Components")
#
# # axs[0].set_xticks(range(1, encoding_dim+1))
# axs[1].set_xticks(range(1, 16))
#
# axs[0].legend()
# axs[1].legend()
#
# plt.savefig("Variational Eagle/Plots/pca_accumulation", bbox_inches="tight")
# plt.show()







no_features_count = [0] * 20

for run in range(1, 26):

    extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_750_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

    pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)

    no_features = len(pca.explained_variance_ratio_)
    no_features_count[no_features] += 1

for i, count in enumerate(no_features_count[7:13]):
    print(i, count)

print()







min_accumulation = []
med_accumulation = []
max_accumulation = []

min_specific = []
med_specific = []
max_specific = []


for components in range(0, 15):

    accumulation_all = []
    specific_all = []

    for run in range(1, 25):

        extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_750_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

        pca = PCA(n_components=encoding_dim, svd_solver="full").fit(extracted_features)
        variance_explained = pca.explained_variance_ratio_

        accumulation = variance_explained[:components+1].sum()
        accumulation_all.append(accumulation*100)

        specific = variance_explained[components]
        specific_all.append(specific*100)

    min_accumulation.append(min(accumulation_all))
    med_accumulation.append(np.median(accumulation_all))
    max_accumulation.append(max(accumulation_all))

    min_specific.append(min(specific_all))
    med_specific.append(np.median(specific_all))
    max_specific.append(max(specific_all))


# # scale font on plots
# default_size = plt.rcParams['font.size']
# plt.rcParams.update({'font.size': default_size * 1.5})
#
# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
#
# axs.plot(range(1, 16), min_specific, color="C1", alpha=0.3)
# axs.plot(range(1, 16), med_specific, color="C1")
# axs.scatter(range(1, 11), med_specific[:10], color="C1", label="Feature Variance")
# axs.plot(range(1, 16), max_specific, color="C1", alpha=0.3)
# axs.fill_between(range(1, 16), min_specific, max_specific, color="C1", alpha=0.075)
#
# axs.plot(range(1, 16), min_accumulation, color="C0", alpha=0.3)
# axs.plot(range(1, 16), med_accumulation, color="C0")
# axs.scatter(range(1, 11), med_accumulation[:10], color="C0", label="Cumulative Variance")
# axs.plot(range(1, 16), max_accumulation, color="C0", alpha=0.3)
# axs.fill_between(range(1, 16), min_accumulation, max_accumulation, color="C0", alpha=0.075)
#
# axs.set_ylabel("Variance Explained (%)")
# axs.set_xlabel("Principal Components")
# axs.set_xticks(list(range(1, 16)))
#
#
#
# print(med_accumulation)
# print(med_specific)
#
# plt.legend()
#
# plt.savefig("Variational Eagle/Plots/variance_explained_lines", bbox_inches="tight")
# plt.show()



# scale font on plots
default_size = plt.rcParams['font.size']
plt.rcParams.update({'font.size': default_size * 1.5})

fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.bar(range(1, 16), med_specific, color="black", alpha=0.4)
error_upper = [max_s - med_s for max_s, med_s in zip(max_specific, med_specific)]
error_lower = [med_s - min_s for med_s, min_s in zip(med_specific, min_specific)]
axs.errorbar(range(1, 16), med_specific, yerr=[error_lower, error_upper], color="black", capsize=3, linestyle="none")

axs.plot(range(1, 16), min_accumulation, color="black", alpha=0.2)
axs.plot(range(1, 16), med_accumulation, color="black")
axs.scatter(range(1, 11), med_accumulation[:10], color="black")
axs.plot(range(1, 16), max_accumulation, color="black", alpha=0.2)
axs.fill_between(range(1, 16), min_accumulation, max_accumulation, color="black", alpha=0.075)

axs.set_ylabel("Variance Explained (%)")
axs.set_xlabel("Principal Components")
axs.set_xticks(list(range(1, 16)))



print(med_accumulation)
print(med_specific)

plt.savefig("Variational Eagle/Plots/variance_explained_bars", bbox_inches="tight")
plt.show()

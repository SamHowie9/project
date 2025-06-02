import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from sklearn.decomposition import PCA



encoding_dim = 35
run = 3
beta = 0.0001
beta_name = "0001"
epochs = 300
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

# add the non parametric properties to the other properties dataframe
all_properties = pd.merge(all_properties, non_parametric_properties, on="GalaxyID")


# find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
bad_fit = all_properties[((all_properties["flag_r"] == 1) |
                          (all_properties["flag_r"] == 4) |
                          (all_properties["flag_r"] == 5) |
                          (all_properties["flag_r"] == 6))].index.tolist()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# reset the index values
all_properties = all_properties.reset_index(drop=True)

# account for the testing dataset
all_properties = all_properties.iloc[:-200]

print(all_properties)












# encoding_dim = 30
#
# correlation_df = pd.DataFrame(columns=["1e-3", "1e-4", "9e-5", "8e-5", "7e-5", "6e-5", "5e-5", "4e-5", "3e-5", "2e-5", "1e-5", "1e-6"])
#
# for feature in range(0, encoding_dim):
#
#     correlation_list = []
#
#     for beta in ["001", "0001", "00009", "00008", "00007", "00006", "00005", "00004", "00003", "00002", "00001", "000001"]:
#
#
#         # load the extracted features
#         # extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
#         extracted_features = np.load("Variational Eagle/Extracted Features/Test/bce_latent_" + str(encoding_dim) + "_beta_" + beta + ".npy")[0]
#         encoding_dim = extracted_features.shape[1]
#         extracted_features_switch = extracted_features.T
#
#         print(extracted_features.shape)
#
#         # remove augmented images
#         extracted_features = extracted_features[:len(all_properties)]
#         extracted_features_switch = extracted_features.T
#
#         # perform pca on the extracted features
#         # pca = PCA(n_components=11).fit(extracted_features)
#         # extracted_features = pca.transform(extracted_features)
#         # extracted_features = extracted_features[:len(all_properties)]
#         # extracted_features_switch = extracted_features.T
#
#
#
#         # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
#         correlation_1 = np.corrcoef(extracted_features_switch[feature], all_properties["n_r"])[0][1]
#         correlation_2 = np.corrcoef(extracted_features_switch[feature], abs(all_properties["n_r"]))[0][1]
#         correlation_3 = np.corrcoef(abs(extracted_features_switch[feature]), all_properties["n_r"])[0][1]
#         correlation_4 = np.corrcoef(abs(extracted_features_switch[feature]), abs(all_properties["n_r"]))[0][1]
#
#         # add the strongest correlation
#         correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))
#
#     correlation_df.loc[len(correlation_df)] = correlation_list
#
# print(correlation_df)
#
#
#
#
# # create string labels for each of the correlations with the extracted feature index
# correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"#{row.name}: {val:.2f}"), axis=1)
#
#
#
#
# print(correlation_text_df)
#
#
# # order each of the columns (remove the number corresponding to each feature)
#
# # order the original dataframe
# correlation_df = pd.DataFrame({col: sorted(correlation_df[col], reverse=True) for col in correlation_df.columns})
#
# # order the annotation dataframe
# for col in correlation_text_df.columns:
#     correlation_text_df[col] = correlation_text_df[col].iloc[
#         correlation_text_df[col].apply(lambda x: float(x.split(': ')[1])).sort_values(ascending=False).index
#     ].values
#
# print(correlation_text_df)
#
#
#
#
#
#
# # fig, axs = plt.subplots(1, 1, figsize=(25, encoding_dim/2))
# fig, axs = plt.subplots(2, 1, figsize=(25, encoding_dim))
#
#
# sns.heatmap(abs(correlation_df), ax=axs, annot=correlation_text_df, fmt="", cmap="Blues", cbar_kws={'label': 'Correlation'})
#
#
# axs.set_yticks([])
# axs.set_ylabel("Extracted Features", fontsize=15)
# axs.xaxis.tick_top() # x axis on top
# axs.xaxis.set_label_position('top')
# axs.tick_params(length=0)
# axs.figure.axes[-1].yaxis.label.set_size(15)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#
#     labels = []
#     # for label in ax.get_xticklabels():
#         # text = label.get_text()
#
#     # label_names = ["β = 1e-3", "β = 1e-4", "β = 1e-5", "β = 1e-6"]
#     # label_names = ["β = 1e-3", "β = 1e-4", "β = 9e-5", "β = 8e-5", "β = 7e-5", "β = 6e-5", "β = 5e-5", "β = 4e-5", "β = 3e-5", "β = 2e-5", "β = 1e-5", "β = 1e-6"]
#     label_names = ["1e-3", "1e-4", "9e-5", "8e-5", "7e-5", "6e-5", "5e-5", "4e-5", "3e-5", "2e-5", "1e-5", "1e-6"]
#
#
#     for text in label_names:
#         labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
#     axs.set_xticklabels(labels, rotation=0, fontsize=15)
#
# wrap_labels(axs, 10)
#
#
# plt.savefig("Variational Eagle/Correlation Plots/Test/beta_sersic_sorted", bbox_inches='tight')
# plt.show()










# encoding_dim = 30


# fig, axs = plt.subplots(1, 1, figsize=(25, encoding_dim/2))
# fig, axs = plt.subplots(2, 1, figsize=(25, encoding_dim))
# fig, axs = plt.subplots(2, 1, figsize=(55, 15))
fig, axs = plt.subplots(2, 1, figsize=(25, 15))





# correlation_df = pd.DataFrame(columns=["1e-3", "1e-4", "9e-5", "8e-5", "7e-5", "6e-5", "5e-5", "4e-5", "3e-5", "2e-5", "1e-5", "1e-6"])
# correlation_df = pd.DataFrame(columns=["20 - 1", "20 - 2", "20 - 3", "25 - 1", "25 - 2", "25 - 3", "30 - 1", "30 - 2", "30 - 3", "35 - 1", "35 - 2", "35 - 3", "40 - 1", "40 - 2", "40 - 3"])
# correlation_df = pd.DataFrame(columns=["25 - 1", "25 - 2", "25 - 3", "26 - 1", "26 - 2", "26 - 3", "27 - 1", "27 - 2", "27 - 3", "28 - 1", "28 - 2", "28 - 3", "29 - 1", "29 - 2", "29 - 3", "30 - 1", "30 - 2", "30 - 3", "31 - 1", "31 - 2", "31 - 3", "32 - 1", "32 - 2", "32 - 3", "33 - 1", "33 - 2", "33 - 3", "34 - 1", "34 - 2", "34 - 3", "35 - 1", "35 - 2", "35 - 3", "36 - 1", "36 - 2", "36 - 3", "37 - 1", "37 - 2", "37 - 3", "38 - 1", "38 - 2", "38 - 3", "39 - 1", "39 - 2", "39 - 3", "40 - 1", "40 - 2", "40 - 3"])
correlation_df = pd.DataFrame(columns=["25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"])

# for feature in range(0, encoding_dim):
for feature in range(0, 15):

    correlation_list = []

    # for beta in ["001", "0001", "00009", "00008", "00007", "00006", "00005", "00004", "00003", "00002", "00001", "000001"]:
    # for encoding_dim, run in [[20, 1], [20, 2], [20, 3], [25, 1], [25, 2], [25, 3], [30, 1], [30, 2], [30, 3], [35, 1], [35, 2], [35, 3], [40, 1], [40, 2], [40, 3]]:
    # for encoding_dim, run in [[25, 1], [25, 2], [25, 3], [26, 1], [26, 2], [25, 3], [27, 1], [27, 2], [27, 3], [28, 1], [28, 2], [28, 3], [29, 1], [29, 2], [29, 3], [30, 1], [30, 2], [30, 3], [31, 1], [31, 2], [31, 3], [32, 1], [32, 2], [32, 3], [33, 1], [33, 2], [33, 3], [34, 1], [34, 2], [34, 3], [35, 1], [35, 2], [35, 3], [36, 1], [36, 2], [36, 3], [37, 1], [37, 2], [37, 3], [38, 1], [38, 2], [38, 3], [39, 1], [39, 2], [39, 3], [40, 1], [40, 2], [40, 3]]:
    for encoding_dim in range(25, 41):

        # load the extracted features
        # extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
        # extracted_features = np.load("Variational Eagle/Extracted Features/Test/bce_latent_" + str(encoding_dim) + "_beta_" + beta + ".npy")[0]
        extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_" + str(run) + ".npy")[0]

        encoding_dim = extracted_features.shape[1]
        extracted_features_switch = extracted_features.T

        print(extracted_features.shape)

        # remove augmented images
        extracted_features = extracted_features[:len(all_properties)]
        extracted_features_switch = extracted_features.T

        # perform pca on the extracted features
        pca = PCA(n_components=15).fit(extracted_features)
        extracted_features = pca.transform(extracted_features)
        extracted_features = extracted_features[:len(all_properties)]
        extracted_features_switch = extracted_features.T



        # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
        correlation_1 = np.corrcoef(extracted_features_switch[feature], all_properties["n_r"])[0][1]
        correlation_2 = np.corrcoef(extracted_features_switch[feature], abs(all_properties["n_r"]))[0][1]
        correlation_3 = np.corrcoef(abs(extracted_features_switch[feature]), all_properties["n_r"])[0][1]
        correlation_4 = np.corrcoef(abs(extracted_features_switch[feature]), abs(all_properties["n_r"]))[0][1]

        # add the strongest correlation
        correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))

    correlation_df.loc[len(correlation_df)] = correlation_list

print(correlation_df)




# create string labels for each of the correlations with the extracted feature index
correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"#{row.name}: {val:.2f}"), axis=1)




print(correlation_text_df)


# order each of the columns (remove the number corresponding to each feature)

# order the original dataframe
correlation_df = pd.DataFrame({col: sorted(correlation_df[col], reverse=True) for col in correlation_df.columns})

# order the annotation dataframe
for col in correlation_text_df.columns:
    correlation_text_df[col] = correlation_text_df[col].iloc[
        correlation_text_df[col].apply(lambda x: float(x.split(': ')[1])).sort_values(ascending=False).index
    ].values

print(correlation_text_df)



sns.heatmap(abs(correlation_df), ax=axs[0], annot=correlation_text_df, fmt="", cmap="Blues", vmax=0.7, cbar_kws={'label': 'Correlation'})

axs[0].set_title("Sersic Index Correlation", fontsize=20, pad=20)
axs[0].set_yticks([])
axs[0].set_ylabel("Extracted Features", fontsize=15, labelpad=10)
axs[0].xaxis.tick_top() # x axis on top
axs[0].xaxis.set_label_position('top')
axs[0].tick_params(length=0)
axs[0].figure.axes[-1].yaxis.label.set_size(15)


def wrap_labels(ax, width, break_long_words=False):

    labels = []
    # for label in ax.get_xticklabels():
        # text = label.get_text()

    # label_names = ["β = 1e-3", "β = 1e-4", "β = 1e-5", "β = 1e-6"]
    # label_names = ["β = 1e-3", "β = 1e-4", "β = 9e-5", "β = 8e-5", "β = 7e-5", "β = 6e-5", "β = 5e-5", "β = 4e-5", "β = 3e-5", "β = 2e-5", "β = 1e-5", "β = 1e-6"]
    # label_names = ["1e-3", "1e-4", "9e-5", "8e-5", "7e-5", "6e-5", "5e-5", "4e-5", "3e-5", "2e-5", "1e-5", "1e-6"]
    # label_names = ["20 - 1", "20 - 2", "20 - 3", "25 - 1", "25 - 2", "25 - 3", "30 - 1", "30 - 2", "30 - 3", "35 - 1", "35 - 2", "35 - 3", "40 - 1", "40 - 2", "40 - 3"]
    # label_names = ["25 - 1", "25 - 2", "25 - 3", "26 - 1", "26 - 2", "26 - 3", "27 - 1", "27 - 2", "27 - 3", "28 - 1", "28 - 2", "28 - 3", "29 - 1", "29 - 2", "29 - 3", "30 - 1", "30 - 2", "30 - 3", "31 - 1", "31 - 2", "31 - 3", "32 - 1", "32 - 2", "32 - 3", "33 - 1", "33 - 2", "33 - 3", "34 - 1", "34 - 2", "34 - 3", "35 - 1", "35 - 2", "35 - 3", "36 - 1", "36 - 2", "36 - 3", "37 - 1", "37 - 2", "37 - 3", "38 - 1", "38 - 2", "38 - 3", "39 - 1", "39 - 2", "39 - 3", "40 - 1", "40 - 2", "40 - 3"]
    label_names = ["25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"]

    for text in label_names:
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    axs[0].set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(axs[0], 10)










# correlation_df = pd.DataFrame(columns=["1e-3", "1e-4", "9e-5", "8e-5", "7e-5", "6e-5", "5e-5", "4e-5", "3e-5", "2e-5", "1e-5", "1e-6"])
# correlation_df = pd.DataFrame(columns=["20 - 1", "20 - 2", "20 - 3", "25 - 1", "25 - 2", "25 - 3", "30 - 1", "30 - 2", "30 - 3", "35 - 1", "35 - 2", "35 - 3", "40 - 1", "40 - 2", "40 - 3"])
# correlation_df = pd.DataFrame(columns=["25 - 1", "25 - 2", "25 - 3", "26 - 1", "26 - 2", "26 - 3", "27 - 1", "27 - 2", "27 - 3", "28 - 1", "28 - 2", "28 - 3", "29 - 1", "29 - 2", "29 - 3", "30 - 1", "30 - 2", "30 - 3", "31 - 1", "31 - 2", "31 - 3", "32 - 1", "32 - 2", "32 - 3", "33 - 1", "33 - 2", "33 - 3", "34 - 1", "34 - 2", "34 - 3", "35 - 1", "35 - 2", "35 - 3", "36 - 1", "36 - 2", "36 - 3", "37 - 1", "37 - 2", "37 - 3", "38 - 1", "38 - 2", "38 - 3", "39 - 1", "39 - 2", "39 - 3", "40 - 1", "40 - 2", "40 - 3"])
correlation_df = pd.DataFrame(columns=["25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"])

# for feature in range(0, encoding_dim):
for feature in range(0, 15):

    correlation_list = []

    # for beta in ["001", "0001", "00009", "00008", "00007", "00006", "00005", "00004", "00003", "00002", "00001", "000001"]:

    # for encoding_dim, run in [[20, 1], [20, 2], [20, 3], [25, 1], [25, 2], [25, 3], [30, 1], [30, 2], [30, 3], [35, 1], [35, 2], [35, 3], [40, 1], [40, 2], [40, 3]]:
    # for encoding_dim, run in [[25, 1], [25, 2], [25, 3], [26, 1], [26, 2], [25, 3], [27, 1], [27, 2], [27, 3], [28, 1], [28, 2], [28, 3], [29, 1], [29, 2], [29, 3], [30, 1], [30, 2], [30, 3], [31, 1], [31, 2], [31, 3], [32, 1], [32, 2], [32, 3], [33, 1], [33, 2], [33, 3], [34, 1], [34, 2], [34, 3], [35, 1], [35, 2], [35, 3], [36, 1], [36, 2], [36, 3], [37, 1], [37, 2], [37, 3], [38, 1], [38, 2], [38, 3], [39, 1], [39, 2], [39, 3], [40, 1], [40, 2], [40, 3]]:
    for encoding_dim in range(25, 41):

        # load the extracted features
        # extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
        # extracted_features = np.load("Variational Eagle/Extracted Features/Test/bce_latent_" + str(encoding_dim) + "_beta_" + beta + ".npy")[0]
        extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_" + str(run) + ".npy")[0]

        encoding_dim = extracted_features.shape[1]
        extracted_features_switch = extracted_features.T

        print(extracted_features.shape)

        # remove augmented images
        extracted_features = extracted_features[:len(all_properties)]
        extracted_features_switch = extracted_features.T

        # perform pca on the extracted features
        pca = PCA(n_components=15).fit(extracted_features)
        extracted_features = pca.transform(extracted_features)
        extracted_features = extracted_features[:len(all_properties)]
        extracted_features_switch = extracted_features.T



        # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
        correlation_1 = np.corrcoef(extracted_features_switch[feature], all_properties["re_r"])[0][1]
        correlation_2 = np.corrcoef(extracted_features_switch[feature], abs(all_properties["re_r"]))[0][1]
        correlation_3 = np.corrcoef(abs(extracted_features_switch[feature]), all_properties["re_r"])[0][1]
        correlation_4 = np.corrcoef(abs(extracted_features_switch[feature]), abs(all_properties["re_r"]))[0][1]

        # add the strongest correlation
        correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))

    correlation_df.loc[len(correlation_df)] = correlation_list

print(correlation_df)




# create string labels for each of the correlations with the extracted feature index
correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"#{row.name}: {val:.2f}"), axis=1)




print(correlation_text_df)


# order each of the columns (remove the number corresponding to each feature)

# order the original dataframe
correlation_df = pd.DataFrame({col: sorted(correlation_df[col], reverse=True) for col in correlation_df.columns})

# order the annotation dataframe
for col in correlation_text_df.columns:
    correlation_text_df[col] = correlation_text_df[col].iloc[
        correlation_text_df[col].apply(lambda x: float(x.split(': ')[1])).sort_values(ascending=False).index
    ].values

print(correlation_text_df)







sns.heatmap(abs(correlation_df), ax=axs[1], annot=correlation_text_df, fmt="", cmap="Blues", vmax=0.7, cbar_kws={'label': 'Correlation'})

axs[1].set_title("Semi-Major Axis Correlation", fontsize=20, pad=20)
axs[1].set_yticks([])
axs[1].set_ylabel("Extracted Features", fontsize=15, labelpad=10)
axs[1].xaxis.tick_top() # x axis on top
axs[1].xaxis.set_label_position('top')
axs[1].tick_params(length=0)
axs[1].figure.axes[-1].yaxis.label.set_size(15)


def wrap_labels(ax, width, break_long_words=False):

    labels = []
    # for label in ax.get_xticklabels():
        # text = label.get_text()

    # label_names = ["β = 1e-3", "β = 1e-4", "β = 1e-5", "β = 1e-6"]
    # label_names = ["β = 1e-3", "β = 1e-4", "β = 9e-5", "β = 8e-5", "β = 7e-5", "β = 6e-5", "β = 5e-5", "β = 4e-5", "β = 3e-5", "β = 2e-5", "β = 1e-5", "β = 1e-6"]
    # label_names = ["1e-3", "1e-4", "9e-5", "8e-5", "7e-5", "6e-5", "5e-5", "4e-5", "3e-5", "2e-5", "1e-5", "1e-6"]
    # label_names = ["20 - 1", "20 - 2", "20 - 3", "25 - 1", "25 - 2", "25 - 3", "30 - 1", "30 - 2", "30 - 3", "35 - 1", "35 - 2", "35 - 3", "40 - 1", "40 - 2", "40 - 3"]
    # label_names = ["25 - 1", "25 - 2", "25 - 3", "26 - 1", "26 - 2", "26 - 3", "27 - 1", "27 - 2", "27 - 3", "28 - 1", "28 - 2", "28 - 3", "29 - 1", "29 - 2", "29 - 3", "30 - 1", "30 - 2", "30 - 3", "31 - 1", "31 - 2", "31 - 3", "32 - 1", "32 - 2", "32 - 3", "33 - 1", "33 - 2", "33 - 3", "34 - 1", "34 - 2", "34 - 3", "35 - 1", "35 - 2", "35 - 3", "36 - 1", "36 - 2", "36 - 3", "37 - 1", "37 - 2", "37 - 3", "38 - 1", "38 - 2", "38 - 3", "39 - 1", "39 - 2", "39 - 3", "40 - 1", "40 - 2", "40 - 3"]
    label_names = ["25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"]

    for text in label_names:
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    axs[1].set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(axs[1], 10)







plt.savefig("Variational Eagle/Correlation Plots/Final/latent_" + str(run), bbox_inches='tight')
plt.show()






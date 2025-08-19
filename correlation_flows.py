import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from sklearn.decomposition import PCA
import dcor




run = 16
encoding_dim = 35
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32








all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")
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













fig, axs = plt.subplots(2, 1, figsize=(35, 35))
# fig, axs = plt.subplots(2, 1, figsize=(35, 15))



# balanced dataset (30, 35, 40, 45, 50)
# run_order = [9, 23] + [4, 5, 7, 11, 12, 20, 24] + [1, 2, 3, 6, 8, 13, 14, 16, 17, 18, 19, 22, 25] + [10, 15, 21]
run_order = [4, 8, 11, 16] + [2, 5, 7, 10, 12, 15, 17, 18, 19, 20, 22, 23] + [1, 3, 6, 9, 13, 14, 21, 24, 25]
# run_order = [9] + [7, 17] + [1, 3, 5, 8, 10, 15, 20, 21, 23, 24] + [2, 4, 6, 11, 12, 13, 14, 16, 18, 19, 22, 25]
# run_order = [2, 3, 10, 11, 20, 22, 24] + [1, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23] + [25]
# run_order = [13, 21] + [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25] + [6, 9, 17]

# face on (35)
# run_order = [2, 4, 7, 17, 21] + [1, 3, 5, 6, 8, 10, 13, 18, 20, 22, 24] + [11, 14, 15, 19, 23] + [9, 12, 16, 25]

# spirals (30, 35)
# run_order = [3, 4, 5, 17, 18] + [1, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25] + [8]
# run_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25] + [20]

# transitional (30, 35)
# run_order = [8] + [10, 12] + [2, 4, 5, 6, 11, 13, 14, 18, 21, 22, 23] + [1, 7, 15, 16, 17, 19, 20, 24, 25] + [3, 9]
# run_order = [3, 7, 14, 20, 22] + [2, 6, 8, 9, 11, 12, 13, 16, 17, 19, 23, 25] + [1, 4, 5, 10, 15, 18, 21, 24]

# ellipticals (30, 35)
# run_order = [1, 2, 10, 13, 16, 18, 21, 24] + [3, 4, 5, 12, 14, 15, 17, 22, 23, 25] + [6, 7, 8, 19, 20] + [9, 11]
# run_order = [6] + [2, 3, 5, 8, 13, 16, 20, 24] + [4, 9, 11, 12, 14, 15, 18, 19, 23, 25] + [1, 7, 10, 17, 21, 22]

run_names = [str(a) for a in run_order]


correlation_df = pd.DataFrame(columns=run_names)


for feature in range(0, encoding_dim):
# for feature in range(0, 13):

    correlation_list = []

    for run in run_order:

        extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
        # extracted_features = np.load("Variational Eagle/Extracted Features/Face/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")



        # pca feature correlation

        # # perform pca on the extracted features
        # pca = PCA(n_components=13).fit(extracted_features)
        # extracted_features = pca.transform(extracted_features)
        #
        # variance = pca.explained_variance_ratio_[feature]
        #
        # if variance >= 0.001:
        #
        #     # # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
        #     # correlation_1 = np.corrcoef(extracted_features.T[feature], all_properties["re_r"])[0][1]
        #     # correlation_2 = np.corrcoef(extracted_features.T[feature], abs(all_properties["re_r"]))[0][1]
        #     # correlation_3 = np.corrcoef(abs(extracted_features.T[feature]), all_properties["re_r"])[0][1]
        #     # correlation_4 = np.corrcoef(abs(extracted_features.T[feature]), abs(all_properties["re_r"]))[0][1]
        #     #
        #     # # add the strongest correlation
        #     # correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))
        #
        #     correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["re_r"])
        #     correlation_list.append(correlation)
        #
        # else:
        #     correlation_list.append(0)



        # latent feature correlation

        correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["re_r"])
        correlation_list.append(correlation)



    correlation_df.loc[len(correlation_df)] = correlation_list

# set index so feature label starts at 1 rather than 0
correlation_df.index = correlation_df.index + 1

print(correlation_df)






# correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"{val:.2f}"), axis=1)
# correlation_text_df = correlation_text_df.replace("0.00", "")
# print(correlation_text_df)



# order each of the columns (remove the number corresponding to each feature)

# create string labels for each of the correlations with the extracted feature index
correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"#{row.name}: {val:.2f}"), axis=1)
print(correlation_text_df)

# order the original dataframe
correlation_df = pd.DataFrame({col: sorted(correlation_df[col], reverse=True) for col in correlation_df.columns})

# order the annotation dataframe
for i, col in enumerate(correlation_text_df.columns):
    print(col)
    print(correlation_text_df[col])
    print(correlation_text_df[col].apply(lambda x: float(x.split(': ')[1])).sort_values(ascending=False).index)
    correlation_text_df[col] = correlation_text_df[col].iloc[
        correlation_text_df[col].apply(lambda x: float(x.split(': ')[1])).sort_values(ascending=False).index
    ].values

print(correlation_text_df)



# sns.heatmap(abs(correlation_df), ax=axs[0], annot=correlation_text_df, fmt="", cmap="Blues", vmax=0.7, cbar_kws={'label': 'Correlation'})
# sns.heatmap(abs(correlation_df), ax=axs[0], annot=True, annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.02, "aspect": 30})
sns.heatmap(abs(correlation_df), ax=axs[0], annot=correlation_text_df, fmt="", annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.015, "aspect": 30})


axs[0].set_title("Sersic Index Correlation", fontsize=20, pad=20)
axs[0].set_ylabel("Extracted Features", fontsize=15)
axs[0].xaxis.tick_top() # x axis on top
axs[0].xaxis.set_label_position('top')
axs[0].tick_params(length=0, labelsize=15)
axs[0].tick_params(axis="y", labelrotation=0)
axs[0].figure.axes[-1].yaxis.label.set_size(15)

colourbar = axs[0].collections[0].colorbar
colourbar.ax.tick_params(labelsize=15)
colourbar.ax.yaxis.label.set_size(15)


def wrap_labels(ax, width, break_long_words=False):

    labels = []
    label_names = run_names

    for text in label_names:
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    axs[0].set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(axs[0], 10)







correlation_df = pd.DataFrame(columns=run_names)


for feature in range(0, encoding_dim):
# for feature in range(0, 13):

    correlation_list = []

    for run in run_order:

        extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
        # extracted_features = np.load("Variational Eagle/Extracted Features/Face/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")



        # pca feature correlation

        # # perform pca on the extracted features
        # pca = PCA(n_components=13).fit(extracted_features)
        # extracted_features = pca.transform(extracted_features)
        #
        # variance = pca.explained_variance_ratio_[feature]
        #
        # if variance >= 0.001:
        #
        #     # # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
        #     # correlation_1 = np.corrcoef(extracted_features.T[feature], all_properties["re_r"])[0][1]
        #     # correlation_2 = np.corrcoef(extracted_features.T[feature], abs(all_properties["re_r"]))[0][1]
        #     # correlation_3 = np.corrcoef(abs(extracted_features.T[feature]), all_properties["re_r"])[0][1]
        #     # correlation_4 = np.corrcoef(abs(extracted_features.T[feature]), abs(all_properties["re_r"]))[0][1]
        #     #
        #     # # add the strongest correlation
        #     # correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))
        #
        #     correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["re_r"])
        #     correlation_list.append(correlation)
        #
        # else:
        #     correlation_list.append(0)


        # latent feature correlation

        correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["re_r"])
        correlation_list.append(correlation)

    correlation_df.loc[len(correlation_df)] = correlation_list


# set index so feature label starts at 1 rather than 0
correlation_df.index = correlation_df.index + 1

print(correlation_df)




# create string labels for each of the correlations with the extracted feature index
correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"{val:.2f}"), axis=1)
correlation_text_df = correlation_text_df.replace("0.00", "")
print(correlation_text_df)




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







# sns.heatmap(abs(correlation_df), ax=axs[1], annot=correlation_text_df, fmt="", cmap="Blues", vmax=0.7, cbar_kws={'label': 'Correlation'})
# sns.heatmap(abs(correlation_df), ax=axs[1], annot=True, annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.02, "aspect": 60})
sns.heatmap(abs(correlation_df), ax=axs[1], annot=correlation_text_df, fmt="", annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.015, "aspect": 30})


axs[1].set_title("Semi-Major Axis Correlation", fontsize=20, pad=20)
axs[1].set_ylabel("Extracted Features", fontsize=15)
axs[1].xaxis.tick_top() # x axis on top
axs[1].xaxis.set_label_position('top')
axs[1].tick_params(length=0, labelsize=15)
axs[1].tick_params(axis="y", labelrotation=0)
axs[1].figure.axes[-1].yaxis.label.set_size(15)

colourbar = axs[1].collections[0].colorbar
colourbar.ax.tick_params(labelsize=15)
colourbar.ax.yaxis.label.set_size(15)


def wrap_labels(ax, width, break_long_words=False):

    labels = []
    label_names = run_names

    for text in label_names:
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    axs[1].set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(axs[1], 10)




# balanced dataset (30, 35, 40, 45, 50)
# cols = [2, 9, 22]
cols = [4, 16]
# cols = [1, 3, 13]
# cols = [7, 24]
# cols = [2, 22]

# face on (35)
# cols = [5, 16, 21]

# spirals (30, 35)
# cols = [5, 24]
# cols = [24]

# transitional (30, 35)
# cols = [1, 3, 14, 23]
# cols = [5, 17]

# ellipticals (35)
# cols = 8, 18, 23
# cols = [1, 9, 19]

ymin, ymax = axs[0].get_ylim()
for col in cols:
    axs[0].vlines(x=col, ymin=ymin, ymax=ymax, color="black", linewidth=2)
    axs[1].vlines(x=col, ymin=ymin, ymax=ymax, color="black", linewidth=2)






plt.savefig("Variational Eagle/Correlation Plots/Normalising Flow/Latent/latent_" + str(encoding_dim) + "_correlation", bbox_inches='tight')
plt.show()






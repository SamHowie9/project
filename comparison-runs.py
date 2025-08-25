import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from sklearn.decomposition import PCA
# import dcor




run = 16
encoding_dim = 35
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32








all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")
print(all_properties)











# # fig, axs = plt.subplots(2, 1, figsize=(50, 35))
# fig, axs = plt.subplots(2, 1, figsize=(35, 15))
#
#
#
# # balanced dataset (30, 35, 40, 45, 50)
# # run_order = [9, 23] + [4, 5, 7, 11, 12, 20, 24] + [1, 2, 3, 6, 8, 13, 14, 16, 17, 18, 19, 22, 25] + [10, 15, 21]
# run_order = [4, 8, 11, 16] + [2, 5, 7, 10, 12, 15, 17, 18, 19, 20, 22, 23] + [1, 3, 6, 9, 13, 14, 21, 24, 25]
# # run_order = [9] + [7, 17] + [1, 3, 5, 8, 10, 15, 20, 21, 23, 24] + [2, 4, 6, 11, 12, 13, 14, 16, 18, 19, 22, 25]
# # run_order = [2, 3, 10, 11, 20, 22, 24] + [1, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23] + [25]
# # run_order = [13, 21] + [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25] + [6, 9, 17]
#
# # face on (35)
# # run_order = [2, 4, 7, 17, 21] + [1, 3, 5, 6, 8, 10, 13, 18, 20, 22, 24] + [11, 14, 15, 19, 23] + [9, 12, 16, 25]
#
# # spirals (30, 35)
# # run_order = [3, 4, 5, 17, 18] + [1, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25] + [8]
# # run_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25] + [20]
#
# # transitional (30, 35)
# # run_order = [8] + [10, 12] + [2, 4, 5, 6, 11, 13, 14, 18, 21, 22, 23] + [1, 7, 15, 16, 17, 19, 20, 24, 25] + [3, 9]
# # run_order = [3, 7, 14, 20, 22] + [2, 6, 8, 9, 11, 12, 13, 16, 17, 19, 23, 25] + [1, 4, 5, 10, 15, 18, 21, 24]
#
# # ellipticals (30, 35)
# # run_order = [1, 2, 10, 13, 16, 18, 21, 24] + [3, 4, 5, 12, 14, 15, 17, 22, 23, 25] + [6, 7, 8, 19, 20] + [9, 11]
# # run_order = [6] + [2, 3, 5, 8, 13, 16, 20, 24] + [4, 9, 11, 12, 14, 15, 18, 19, 23, 25] + [1, 7, 10, 17, 21, 22]
#
# run_names = [str(a) for a in run_order]
#
#
# correlation_df = pd.DataFrame(columns=run_names)
#
#
# # for feature in range(0, encoding_dim):
# for feature in range(0, 13):
#
#     correlation_list = []
#
#     for run in run_order:
#
#         extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
#         # extracted_features = np.load("Variational Eagle/Extracted Features/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
#
#
#
#         # pca feature correlation
#
#         # perform pca on the extracted features
#         pca = PCA(n_components=13, svd_solver="full").fit(extracted_features)
#         extracted_features = pca.transform(extracted_features)
#
#         variance = pca.explained_variance_ratio_[feature]
#
#         if variance >= 0.001:
#
#             # # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
#             # correlation_1 = np.corrcoef(extracted_features.T[feature], all_properties["n_r"])[0][1]
#             # correlation_2 = np.corrcoef(extracted_features.T[feature], abs(all_properties["n_r"]))[0][1]
#             # correlation_3 = np.corrcoef(abs(extracted_features.T[feature]), all_properties["n_r"])[0][1]
#             # correlation_4 = np.corrcoef(abs(extracted_features.T[feature]), abs(all_properties["n_r"]))[0][1]
#             #
#             # # add the strongest correlation
#             # correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))
#
#
#             # calculate the dcor value
#             correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["n_r"])
#             correlation_list.append(correlation)
#
#         else:
#             correlation_list.append(0)
#
#
#
#         # latent feature correlation
#
#         # correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["n_r"])
#         # correlation_list.append(correlation)
#
#
#
#     correlation_df.loc[len(correlation_df)] = correlation_list
#
# # set index so feature label starts at 1 rather than 0
# correlation_df.index = correlation_df.index + 1
#
# print(correlation_df)
#
# # get the min, median, max correlation for each component
# selected_runs = ["2", "5", "7", "10", "12", "15", "17", "18", "19", "20", "22", "23"]
# sersic_correlation = correlation_df.loc[:11, selected_runs].apply(lambda row: [row.min(), row.median(), row.max()], axis=1).tolist()
# sersic_correlation = np.array(sersic_correlation)
#
#
#
#
#
# correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"{val:.2f}"), axis=1)
# correlation_text_df = correlation_text_df.replace("0.00", "")
# print(correlation_text_df)
#
#
#
# # order each of the columns (remove the number corresponding to each feature)
#
# # # create string labels for each of the correlations with the extracted feature index
# # correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"#{row.name}: {val:.2f}"), axis=1)
# # print(correlation_text_df)
# #
# # # order the original dataframe
# # correlation_df = pd.DataFrame({col: sorted(correlation_df[col], reverse=True) for col in correlation_df.columns})
# #
# # # order the annotation dataframe
# # for i, col in enumerate(correlation_text_df.columns):
# #     correlation_text_df[col] = correlation_text_df[col].loc[
# #         correlation_text_df[col].apply(lambda x: float(x.split(': ')[1])).sort_values(ascending=False).index
# #     ].values
# #
# # print(correlation_text_df)
#
#
#
# # sns.heatmap(abs(correlation_df), ax=axs[0], annot=correlation_text_df, fmt="", cmap="Blues", vmax=0.7, cbar_kws={'label': 'Correlation'})
# # sns.heatmap(abs(correlation_df), ax=axs[0], annot=True, annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.02, "aspect": 30})
# sns.heatmap(abs(correlation_df), ax=axs[0], annot=correlation_text_df, fmt="", annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.015, "aspect": 30})
#
#
# axs[0].set_title("Sersic Index Correlation", fontsize=20, pad=20)
# axs[0].set_ylabel("Extracted Features", fontsize=15)
# axs[0].xaxis.tick_top() # x axis on top
# axs[0].xaxis.set_label_position('top')
# axs[0].tick_params(length=0, labelsize=15)
# axs[0].tick_params(axis="y", labelrotation=0)
# axs[0].figure.axes[-1].yaxis.label.set_size(15)
#
# colourbar = axs[0].collections[0].colorbar
# colourbar.ax.tick_params(labelsize=15)
# colourbar.ax.yaxis.label.set_size(15)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#
#     labels = []
#     label_names = run_names
#
#     for text in label_names:
#         labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
#     axs[0].set_xticklabels(labels, rotation=0, fontsize=15)
#
# wrap_labels(axs[0], 10)
#
#
#
#
#
#
#
# correlation_df = pd.DataFrame(columns=run_names)
#
#
# # for feature in range(0, encoding_dim):
# for feature in range(0, 13):
#
#     correlation_list = []
#
#     for run in run_order:
#
#         extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
#         # extracted_features = np.load("Variational Eagle/Extracted Features/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
#
#
#
#         # pca feature correlation
#
#         # perform pca on the extracted features
#         pca = PCA(n_components=13, svd_solver="full").fit(extracted_features)
#         extracted_features = pca.transform(extracted_features)
#
#         variance = pca.explained_variance_ratio_[feature]
#
#         if variance >= 0.001:
#
#             # # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
#             # correlation_1 = np.corrcoef(extracted_features.T[feature], all_properties["rhalf_ellip"])[0][1]
#             # correlation_2 = np.corrcoef(extracted_features.T[feature], abs(all_properties["rhalf_ellip"]))[0][1]
#             # correlation_3 = np.corrcoef(abs(extracted_features.T[feature]), all_properties["rhalf_ellip"])[0][1]
#             # correlation_4 = np.corrcoef(abs(extracted_features.T[feature]), abs(all_properties["rhalf_ellip"]))[0][1]
#             #
#             # # add the strongest correlation
#             # correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))
#
#             # calculate the dcor value
#             correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["rhalf_ellip"])
#             correlation_list.append(correlation)
#
#         else:
#             correlation_list.append(0)
#
#
#         # latent feature correlation
#
#         # correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties["rhalf_ellip"])
#         # correlation_list.append(correlation)
#
#     correlation_df.loc[len(correlation_df)] = correlation_list
#
#
# # set index so feature label starts at 1 rather than 0
# correlation_df.index = correlation_df.index + 1
#
# print(correlation_df)
#
# # get the min, median, max correlation for each component
# selected_runs = ["2", "5", "7", "10", "12", "15", "17", "18", "19", "20", "22", "23"]
# radius_correlation = correlation_df.loc[:11, selected_runs].apply(lambda row: [row.min(), row.median(), row.max()], axis=1).tolist()
# radius_correlation = np.array(radius_correlation)
#
#
#
#
# # create string labels for each of the correlations with the extracted feature index
# correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"{val:.2f}"), axis=1)
# correlation_text_df = correlation_text_df.replace("0.00", "")
# print(correlation_text_df)
#
#
#
#
#
# # order each of the columns (remove the number corresponding to each feature)
#
# # # create string labels for each of the correlations with the extracted feature index
# # correlation_text_df = correlation_df.apply(lambda row: row.map(lambda val: f"#{row.name}: {val:.2f}"), axis=1)
# # print(correlation_text_df)
# #
# # # order the original dataframe
# # correlation_df = pd.DataFrame({col: sorted(correlation_df[col], reverse=True) for col in correlation_df.columns})
# #
# # # order the annotation dataframe
# # for col in correlation_text_df.columns:
# #     correlation_text_df[col] = correlation_text_df[col].loc[
# #         correlation_text_df[col].apply(lambda x: float(x.split(': ')[1])).sort_values(ascending=False).index
# #     ].values
# #
# # print(correlation_text_df)
#
#
#
#
#
#
#
# # sns.heatmap(abs(correlation_df), ax=axs[1], annot=correlation_text_df, fmt="", cmap="Blues", vmax=0.7, cbar_kws={'label': 'Correlation'})
# # sns.heatmap(abs(correlation_df), ax=axs[1], annot=True, annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.02, "aspect": 60})
# sns.heatmap(abs(correlation_df), ax=axs[1], annot=correlation_text_df, fmt="", annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.015, "aspect": 30})
#
#
# axs[1].set_title("Half-Light Radius", fontsize=20, pad=20)
# axs[1].set_ylabel("Extracted Features", fontsize=15)
# axs[1].xaxis.tick_top() # x axis on top
# axs[1].xaxis.set_label_position('top')
# axs[1].tick_params(length=0, labelsize=15)
# axs[1].tick_params(axis="y", labelrotation=0)
# axs[1].figure.axes[-1].yaxis.label.set_size(15)
#
# colourbar = axs[1].collections[0].colorbar
# colourbar.ax.tick_params(labelsize=15)
# colourbar.ax.yaxis.label.set_size(15)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#
#     labels = []
#     label_names = run_names
#
#     for text in label_names:
#         labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
#     axs[1].set_xticklabels(labels, rotation=0, fontsize=15)
#
# wrap_labels(axs[1], 10)
#
#
#
#
# # balanced dataset (30, 35, 40, 45, 50)
# # cols = [2, 9, 22]
# cols = [4, 16]
# # cols = [1, 3, 13]
# # cols = [7, 24]
# # cols = [2, 22]
#
# # face on (35)
# # cols = [5, 16, 21]
#
# # spirals (30, 35)
# # cols = [5, 24]
# # cols = [24]
#
# # transitional (30, 35)
# # cols = [1, 3, 14, 23]
# # cols = [5, 17]
#
# # ellipticals (35)
# # cols = 8, 18, 23
# # cols = [1, 9, 19]
#
# ymin, ymax = axs[0].get_ylim()
# for col in cols:
#     axs[0].vlines(x=col, ymin=ymin, ymax=ymax, color="black", linewidth=2)
#     axs[1].vlines(x=col, ymin=ymin, ymax=ymax, color="black", linewidth=2)
#
#
#
#
#
# # plt.savefig("Variational Eagle/Correlation Plots/Normalising Flow Balanced/Latent/run_comparison_" + str(encoding_dim) + "_features", bbox_inches='tight')
# plt.show(block=False)
# plt.close()















# print(sersic_correlation.shape)
# print(sersic_correlation.T[1].shape)
#
# radius_error = [radius_correlation.T[1]-radius_correlation.T[0], radius_correlation.T[2]-radius_correlation.T[1]]
# sersic_error = [sersic_correlation.T[1]-sersic_correlation.T[0], sersic_correlation.T[2]-sersic_correlation.T[1]]
#
# default_x = np.array(range(1, 12))
# radius_x = default_x - 0.1
# sersic_x = default_x + 0.1
#
# fig, axs = plt.subplots(1, 1, figsize=(20, 10))
#
# # axs.scatter(x=range(1, 12), y=sersic_correlation.T[1])
# axs.bar(radius_x, radius_correlation.T[1], label="Half-Light Radius", width=0.2)
# axs.errorbar(x=radius_x, y=radius_correlation.T[1], yerr=radius_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=radius_x, y=radius_correlation.T[1], yerr=radius_error, fmt="o", label="Half-Light Radius", markersize=10, lw=2)
#
# axs.bar(sersic_x, sersic_correlation.T[1], yerr=sersic_error, label="Sersic Index", width=0.2)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)
#
# axs.set_ylabel("dCor", fontsize=20)
# axs.set_xlabel("Principal Components", fontsize=20)
# axs.set_xticks(range(1, 12))
# axs.tick_params(axis="both", labelsize=20)
# axs.legend(fontsize=20)
#
#
#
# plt.savefig("Variational Eagle/Correlation Plots/Normalising Flow Balanced/PCA/structure_correlation_run_comparison", bbox_inches='tight')
# plt.show()















def get_property_correlation(property):

    selected_runs = [2, 5, 7, 10, 12, 15, 17, 18, 19, 20, 22, 23]
    run_names = [str(a) for a in selected_runs]

    correlation_df = pd.DataFrame(columns=run_names)

    for feature in range(0, 11):

        correlation_list = []

        for run in selected_runs:

            extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
            # extracted_features = np.load("Variational Eagle/Extracted Features/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

            # pca feature correlation

            # perform pca on the extracted features
            pca = PCA(n_components=11, svd_solver="full").fit(extracted_features)
            extracted_features = pca.transform(extracted_features)


            # # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
            correlation = np.corrcoef(extracted_features.T[feature], all_properties[property])[0][1]
            correlation_list.append(abs(correlation))


            # calculate the dcor value
            # correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties[property])
            # correlation_list.append(correlation)

        correlation_df.loc[len(correlation_df)] = correlation_list

    # set index so feature label starts at 1 rather than 0
    correlation_df.index = correlation_df.index + 1

    # get the min, median, max correlation for each component
    correlation = correlation_df.apply(lambda row: [row.min(), row.median(), row.max()], axis=1).tolist()
    correlation = np.array(correlation)
    return correlation




radius_correlation = get_property_correlation(property="rhalf_ellip")
radius_error = [radius_correlation.T[1]-radius_correlation.T[0], radius_correlation.T[2]-radius_correlation.T[1]]

sersic_correlation = get_property_correlation(property="n_r")
sersic_error = [sersic_correlation.T[1]-sersic_correlation.T[0], sersic_correlation.T[2]-sersic_correlation.T[1]]

axis_ratio_correlation = get_property_correlation(property="q_r")
axis_ratio_error = [axis_ratio_correlation.T[1]-axis_ratio_correlation.T[0], axis_ratio_correlation.T[2]-axis_ratio_correlation.T[1]]

position_angle_correlation = get_property_correlation(property="pa_r")
position_angle_error = [position_angle_correlation.T[1]-position_angle_correlation.T[0], position_angle_correlation.T[2]-position_angle_correlation.T[1]]

concentration_correlation = get_property_correlation(property="concentration")
concentration_error = [concentration_correlation.T[1]-concentration_correlation.T[0], concentration_correlation.T[2]-concentration_correlation.T[1]]

asymmetry_correlation = get_property_correlation(property="asymmetry")
asymmetry_error = [asymmetry_correlation.T[1]-asymmetry_correlation.T[0], asymmetry_correlation.T[2]-asymmetry_correlation.T[1]]

smoothness_correlation = get_property_correlation(property="smoothness")
smoothness_error = [smoothness_correlation.T[1]-smoothness_correlation.T[0], smoothness_correlation.T[2]-smoothness_correlation.T[1]]






fig, axs = plt.subplots(1, 1, figsize=(20, 10))

default_x = np.array(range(1, 12))
radius_x = default_x - 0.1
sersic_x = default_x + 0.1

# axs.scatter(x=range(1, 12), y=sersic_correlation.T[1])
axs.bar(radius_x, radius_correlation.T[1], label="Half-Light Radius", width=0.2, yerr=radius_error, capsize=2, lw=1)
# axs.errorbar(x=radius_x, y=radius_correlation.T[1], yerr=radius_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# axs.errorbar(x=radius_x, y=radius_correlation.T[1], yerr=radius_error, fmt="o", label="Half-Light Radius", markersize=10, lw=2)

axs.bar(sersic_x, sersic_correlation.T[1], label="Sersic Index", width=0.2, yerr=sersic_error, capsize=2, lw=1)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)

axs.set_ylabel("dCor", fontsize=20)
axs.set_xlabel("Principal Components", fontsize=20)
axs.set_xticks(range(1, 12))
axs.tick_params(axis="both", labelsize=20)
axs.legend(fontsize=20)



plt.savefig("Variational Eagle/Correlation Plots/Normalising Flow Balanced/PCA/sersic_radius_correlation_run_comparison_bars", bbox_inches='tight')
plt.show()








# fig, axs = plt.subplots(1, 1, figsize=(20, 10))
#
# radius, sersic, q, pa, c, a, s
#
# default_x = np.array(range(1, 12))
#
# radius_x = default_x - 0.3
# sersic_x = default_x - 0.2
# axis_ratio_x = default_x - 0.1
# position_angle_x = default_x
# concentration_x = default_x + 0.1
# asymmetry_x = default_x + 0.2
# smoothness_x = default_x + 0.03
#
#
# # axs.scatter(x=range(1, 12), y=sersic_correlation.T[1])
# axs.bar(radius_x, radius_correlation.T[1], label="Half-Light Radius", width=0.1)
# axs.errorbar(x=radius_x, y=radius_correlation.T[1], yerr=radius_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=radius_x, y=radius_correlation.T[1], yerr=radius_error, fmt="o", label="Half-Light Radius", markersize=10, lw=2)
#
# axs.bar(sersic_x, sersic_correlation.T[1], yerr=sersic_error, label="Sersic Index", width=0.1)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)
#
# axs.bar(axis_ratio_x, axis_ratio_correlation.T[1], yerr=axis_ratio_error, label="Axis Ratio", width=0.1)
# axs.errorbar(x=axis_ratio_x, y=axis_ratio_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)
#
# axs.bar(sersic_x, sersic_correlation.T[1], yerr=sersic_error, label="Sersic Index", width=0.1)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)
#
# axs.bar(sersic_x, sersic_correlation.T[1], yerr=sersic_error, label="Sersic Index", width=0.1)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)
#
# axs.bar(sersic_x, sersic_correlation.T[1], yerr=sersic_error, label="Sersic Index", width=0.1)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)
#
# axs.bar(sersic_x, sersic_correlation.T[1], yerr=sersic_error, label="Sersic Index", width=0.1)
# axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", color="black", capsize=2, markersize=5, lw=1)
# # axs.errorbar(x=sersic_x, y=sersic_correlation.T[1], yerr=sersic_error, fmt="o", label="Sersic Index", markersize=10, lw=2)
#
# axs.set_ylabel("dCor", fontsize=20)
# axs.set_xlabel("Principal Components", fontsize=20)
# axs.set_xticks(range(1, 12))
# axs.tick_params(axis="both", labelsize=20)
# axs.legend(fontsize=20)
#
#
#
# plt.savefig("Variational Eagle/Correlation Plots/Normalising Flow Balanced/PCA/structure_correlation_run_comparison_bars", bbox_inches='tight')
# plt.show()

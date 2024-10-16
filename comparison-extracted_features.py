import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from yellowbrick.cluster import KElbowVisualizer
from scipy.optimize import curve_fit
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import NearestCentroid


plt.style.use("default")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


encoding_dim = 10


extracted_features = np.load("Variational Eagle/Extracted Features/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]
extracted_features_switch = np.flipud(np.rot90(extracted_features))



# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")



bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
print(bad_fit)

for i, galaxy in enumerate(bad_fit):
    extracted_features = np.delete(extracted_features, galaxy-i, 0)
    all_properties = all_properties.drop(galaxy, axis=0)

print(all_properties[all_properties["flag_r"] == 4])
print(all_properties[all_properties["flag_r"] == 1])
print(all_properties[all_properties["flag_r"] == 5])

extracted_features_switch = extracted_features.T



# model = AgglomerativeClustering()
#
# visualizer = KElbowVisualizer(model, k=(1, 40), timings=False)
#
# visualizer.fit(extracted_features)        # Fit the data to the visualizer
# visualizer.show()







structure_correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude"])
physical_correlation_df = pd.DataFrame(columns=["Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])
correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])

for feature in range(0, len(extracted_features_switch)):

    # create a list to contain the correlation between that feature and each property
    structure_correlation_list = []
    physical_correlation_list = []

    correlation_list = []

    # loop through each property
    # for gal_property in range(1, len(structure_properties.columns)):
    for gal_property in range(1, len(all_properties.columns)):

        if gal_property == 6:
            continue

        # calculate the correlation between that extracted feature and that property
        # structure_correlation = np.corrcoef(extracted_features_switch[feature], abs(structure_properties.iloc[:, gal_property]))[0][1]
        # structure_correlation_list.append(structure_correlation)

        correlation_1 = np.corrcoef(extracted_features_switch[feature], all_properties.iloc[:, gal_property])[0][1]
        correlation_2 = np.corrcoef(extracted_features_switch[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
        correlation_3 = np.corrcoef(abs(extracted_features_switch[feature]), all_properties.iloc[:, gal_property])[0][1]
        correlation_4 = np.corrcoef(abs(extracted_features_switch[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]

        if gal_property == 1 and feature == 21:
            print(correlation_1, correlation_2, correlation_3, correlation_4)

        correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))

        # # linear, quadratic and cubic fitting
        # linear = np.polyfit(x=extracted_features_switch[feature], y=all_properties.iloc[:, gal_property], deg=1)
        # quadratic = np.polyfit(x=extracted_features_switch[feature], y=all_properties.iloc[:, gal_property], deg=2)
        # cubic = np.polyfit(x=extracted_features_switch[feature], y=all_properties.iloc[:, gal_property], deg=3)
        #
        # # evaluate these polynomials
        # val_linear = np.polyval(linear, extracted_features_switch[feature])
        # val_quadratic = np.polyval(quadratic, extracted_features_switch[feature])
        # val_cubic = np.polyval(cubic, extracted_features_switch[feature])
        #
        # # mean squared error between polynomial fits and data
        # mse_linear = np.mean(np.square(all_properties.iloc[:, gal_property] - val_linear))
        # mse_quadratic = np.mean(np.square(all_properties.iloc[:, gal_property] - val_quadratic))
        # mse_cubic = np.mean(np.square(all_properties.iloc[:, gal_property] - val_cubic))
        #
        # best_fit = min(mse_linear, mse_quadratic, mse_cubic)
        #
        # correlation_list.append(best_fit)



    # for gal_property in range(1, len(physical_properties.columns)):
    #
    #     # calculate the correlation between that extracted feature and that property
    #     physical_correlation = np.corrcoef(extracted_features_switch[feature], physical_properties.iloc[:, gal_property])[0][1]
    #     physical_correlation_list.append(physical_correlation)

    # # add the correlation of that feature to the main dataframe
    # structure_correlation_df.loc[len(structure_correlation_df)] = structure_correlation_list
    # physical_correlation_df.loc[len(physical_correlation_df)] = physical_correlation_list
    # correlation_df.loc[len(correlation_df)] = structure_correlation_list + physical_correlation_list

    # print(correlation_list)
    #
    correlation_df.loc[len(correlation_df)] = correlation_list




# print(structure_correlation_df)
# print(physical_correlation_df)

# print(correlation_df)





# # set the figure size
# plt.figure(figsize=(12, 16))
#
# sns.set(font_scale=1.5)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df[["Sersic Index", "Position Angle", "Axis Ratio"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation Coefficient'})
#
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=22, labelpad=10)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
#
# ax.figure.axes[-1].yaxis.label.set_size(22)
# ax.figure.axes[-1].yaxis.labelpad = 15
#
#
# # colorbar = plt.colorbar(ax)
# # colorbar.setlabel("Correlation Coefficient", fontsize=22, labelpad=10)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=22)
#
# wrap_labels(ax, 10)
#
#
#
# plt.savefig("Variational Eagle/Correlation Plots/" + str(encoding_dim) + "_feature_structure_measurement_correlation_1")
# plt.show()





# # set the figure size
# plt.figure(figsize=(20, 16))
#
# sns.set(font_scale=1.5)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df[["Semi - Major Axis", "Stellar Age", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation Coefficient'})
#
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=22)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
#
# ax.figure.axes[-1].yaxis.label.set_size(22)
# ax.figure.axes[-1].yaxis.labelpad = 15
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=22)
#
# wrap_labels(ax, 10)
#
#
#
# plt.savefig("Correlation Plots Rand/" + str(encoding_dim) + "_feature_physical_property_correlation_1")
# plt.show()







selected_properties = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass", "Stellar Age", "Star Formation Rate"]

print(correlation_df)


# set the figure size
plt.figure(figsize=(20, encoding_dim))

# sns.set(font_scale = 2)

# ["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"]

# plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df[["Sersic Index", "Semi - Major Axis", "AB Magnitude", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
ax = sns.heatmap(abs(correlation_df[selected_properties]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})



plt.yticks(rotation=0)
plt.ylabel("Extracted Features", fontsize=15)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(length=0)
ax.figure.axes[-1].yaxis.label.set_size(15)


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(ax, 10)



plt.savefig("Variational Eagle/Correlation Plots/" + str(encoding_dim) + "_feature_all_property_correlation_1_abs", bbox_inches='tight')
plt.show()


# plt.figure(figsize=(10,10))
# plt.scatter(x=extracted_features_switch[9], y=all_properties["q_r"])
# plt.show()





# properties = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass", "Stellar Age", "Star Formation Rate"]
properties = ["n_r", "pa_r", "q_r", "re_r", "InitialMassWeightedStellarAge", "StarFormationRate", "MassType_Star", "MassType_DM", "MassType_BH"]


all_properties = all_properties[["n_r", "pa_r", "q_r", "re_r", "InitialMassWeightedStellarAge", "StarFormationRate", "MassType_Star", "MassType_DM", "MassType_BH"]]

print(all_properties)

property_labels = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "Stellar Age", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]

fig, axs = plt.subplots(encoding_dim, len(properties), figsize=(40, (encoding_dim * 4)))
# fig, axs = plt.subplots(19, len(properties), figsize=(40, 70))

# sns.set(font_scale=10)

for i, property in enumerate(properties):

    axs[0][i].set_title(property_labels[i], fontsize=20)

    # for feature in range(0, 19):
    for feature in range(0, encoding_dim):

        axs[feature][i].scatter(x=extracted_features_switch[feature], y=all_properties[property], s=0.5)
        # axs[feature][i].scatter(x=extracted_features_switch[feature+19], y=all_properties[property], s=0.5)

        # sns.kdeplot(data=all_properties, x=extracted_features_switch[feature], y=all_properties[property], gridsize=200)

        axs[feature][i].set_xlabel("Feature " + str(feature), fontsize=12)
        # axs[feature][i].set_xlabel("Feature " + str(feature+19), fontsize=12)
        axs[feature][i].set_ylabel(None)
        axs[feature][i].tick_params(labelsize=12)

plt.savefig("Variational Eagle/Correlation Plots/scatter_" + str(encoding_dim) + "_feature_all_property_correlation_p1", bbox_inches='tight')
# plt.savefig("Variational Eagle/Correlation Plots/scatter_" + str(encoding_dim) + "_feature_all_property_correlation_p2", bbox_inches='tight')
plt.show()





# # # sns.kdeplot(data=all_properties, x=extracted_features_switch[4], y=all_properties["MassType_Star"], levels=200, fill=True, cmap="mako")
# #
# # plt.hist2d(x=extracted_features_switch[4], y=all_properties["MassType_Star"], bins=50, cmap="jet")
# #
# # plt.show()



# def fit_1(x, a, b):
#     return a*x + b
#
# def fit_2(x, a, b, c):
#     return a*x*x + b*x + c
#
# def fit_3(x, a, b, c, d):
#     return a*x*x*x + b*x*x + c*x + d
#
# def fit_4(x, a, b, c, d, e):
#     return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e




# params, covarience = curve_fit(fit_2, extracted_features_switch[3], all_properties["n_r"])
# a, b, c = params
# x_fit_1 = np.linspace(-5, 25, 100).tolist()
# y_fit_1 = []
# for x in x_fit_1:
#     y_fit_1.append(fit_2(x, a, b, c))
#
# params, covarience = curve_fit(fit_2, extracted_features_switch[12], all_properties["n_r"])
# a, b, c = params
# x_fit_2 = np.linspace(-25, 3, 100).tolist()
# y_fit_2 = []
# for x in x_fit_2:
#     y_fit_2.append(fit_2(x, a, b, c))
#
# params, covarience = curve_fit(fit_2, extracted_features_switch[20], all_properties["n_r"])
# a, b, c = params
# x_fit_3 = np.linspace(-10, 10, 100).tolist()
# y_fit_3 = []
# for x in x_fit_3:
#     y_fit_3.append(fit_2(x, a, b, c))
#
#
# fig, axs = plt.subplots(1, 3, figsize=(25, 5))
#
# axs[0].scatter(x=extracted_features_switch[3], y=all_properties["n_r"], s=3, alpha=0.5)
# axs[0].plot(x_fit_1, y_fit_1, c="black")
# axs[0].set_xlabel("Feature 3", fontsize=20)
# axs[0].set_ylabel("Sersic Index", fontsize=20)
# axs[0].tick_params(labelsize=20)
#
# axs[1].scatter(x=extracted_features_switch[12], y=all_properties["n_r"], s=3, alpha=0.5)
# axs[1].plot(x_fit_2, y_fit_2, c="black")
# axs[1].set_xlabel("Feature 12", fontsize=20)
# axs[1].set_ylabel("Sersic Index", fontsize=20)
# axs[1].tick_params(labelsize=20)
#
# axs[2].scatter(x=extracted_features_switch[20], y=all_properties["n_r"], s=3, alpha=0.5)
# axs[2].plot(x_fit_3, y_fit_3, c="black")
# axs[2].set_xlabel("Feature 20", fontsize=20)
# axs[2].set_ylabel("Sersic Index", fontsize=20)
# axs[2].tick_params(labelsize=20)
#
# plt.savefig("Plots/Sersic_scatter", bbox_inches='tight')
# plt.show()







# params, covarience = curve_fit(fit_3, all_properties["pa_r"], extracted_features_switch[1])
# a, b, c, d = params
# x_fit_1 = np.linspace(-90, 90, 100).tolist()
# y_fit_1 = []
# for x in x_fit_1:
#     y_fit_1.append(fit_3(x, a, b, c, d))
#
# params, covarience = curve_fit(fit_4, all_properties["pa_r"], extracted_features_switch[8])
# a, b, c, d, e = params
# x_fit_2 = np.linspace(-90, 90, 100).tolist()
# y_fit_2 = []
# for x in x_fit_2:
#     y_fit_2.append(fit_4(x, a, b, c, d, e))
#
#
# params, covarience = curve_fit(fit_2, all_properties["pa_r"], extracted_features_switch[14])
# a, b, c = params
# x_fit_3 = np.linspace(-90, 90, 100).tolist()
# y_fit_3 = []
# for x in x_fit_3:
#     y_fit_3.append(fit_2(x, a, b, c))
#
#
# fig, axs = plt.subplots(1, 3, figsize=(25, 5))
#
# axs[0].scatter(x=extracted_features_switch[1], y=all_properties["pa_r"], s=3, alpha=0.5)
# axs[0].plot(y_fit_1, x_fit_1, c="black")
# axs[0].set_xlabel("Feature 1", fontsize=20)
# axs[0].set_ylabel("Position Angle", fontsize=20, labelpad=-10)
# axs[0].set_yticks([-90, 0, 90])
# axs[0].tick_params(labelsize=20)
#
# axs[1].scatter(x=extracted_features_switch[8], y=all_properties["pa_r"], s=3, alpha=0.5)
# axs[1].plot(y_fit_2, x_fit_2, c="black")
# axs[1].set_xlabel("Feature 8", fontsize=20)
# axs[1].set_ylabel("Position Angle", fontsize=20, labelpad=-10)
# axs[1].set_yticks([-90, 0, 90])
# axs[1].tick_params(labelsize=20)
#
# axs[2].scatter(x=extracted_features_switch[14], y=all_properties["pa_r"], s=3, alpha=0.5)
# axs[2].plot(y_fit_3, x_fit_3, c="black")
# axs[2].set_xlabel("Feature 14", fontsize=20)
# axs[2].set_ylabel("Position Angle", fontsize=20, labelpad=-10)
# axs[2].set_yticks([-90, 0, 90])
# axs[2].tick_params(labelsize=20)
#
# plt.savefig("Plots/position_angle_scatter", bbox_inches='tight')
# plt.show()





# params, covarience = curve_fit(fit_2, abs(extracted_features_switch[1]), all_properties["q_r"])
# a, b, c = params
# x_fit_1 = np.linspace(0, 11, 100).tolist()
# neg_x_fit_1 = [-x for x in x_fit_1]
# y_fit_1 = []
# for x in x_fit_1:
#     y_fit_1.append(fit_2(x, a, b, c))
#
#
# params, covarience = curve_fit(fit_2, abs(extracted_features_switch[8]), all_properties["q_r"])
# a, b, c = params
# x_fit_2 = np.linspace(0, 11, 100).tolist()
# neg_x_fit_2 = [-x for x in x_fit_1]
# y_fit_2 = []
# for x in x_fit_1:
#     y_fit_2.append(fit_2(x, a, b, c))
#
#
# fig, axs = plt.subplots(1, 2, figsize=(17, 5))
#
# axs[0].scatter(x=extracted_features_switch[1], y=all_properties["q_r"], s=3, alpha=0.75)
# axs[0].plot(x_fit_1, y_fit_1, c="black")
# axs[0].plot(neg_x_fit_1, y_fit_1, c="black")
# axs[0].set_xlabel("Feature 1", fontsize=20)
# axs[0].set_ylabel("Axis Ratio", fontsize=20)
# axs[0].tick_params(labelsize=20)
#
# axs[1].scatter(x=extracted_features_switch[8], y=all_properties["q_r"], s=3, alpha=0.75)
# axs[1].plot(x_fit_2, y_fit_2, c="black")
# axs[1].plot(neg_x_fit_2, y_fit_2, c="black")
# axs[1].set_xlabel("Feature 8", fontsize=20)
# axs[1].set_ylabel("Axis Ratio", fontsize=20)
# axs[1].tick_params(labelsize=20)
#
# plt.savefig("Plots/axis_ratio_scatter", bbox_inches='tight')
# plt.show()






# params, covarience = curve_fit(fit_2, extracted_features_switch[7], all_properties["StarFormationRate"])
# a, b, c = params
# x_fit_1 = np.linspace(-20, -3, 100).tolist()
# y_fit_1 = []
# for x in x_fit_1:
#     y_fit_1.append(fit_2(x, a, b, c))
#
# params, covarience = curve_fit(fit_2, extracted_features_switch[12], all_properties["StarFormationRate"])
# a, b, c = params
# x_fit_2 = np.linspace(-20, 0, 100).tolist()
# y_fit_2 = []
# for x in x_fit_1:
#     y_fit_2.append(fit_2(x, a, b, c))
#
# params, covarience = curve_fit(fit_2, extracted_features_switch[7], all_properties["re_r"])
# a, b, c = params
# x_fit_3 = np.linspace(-20, -3, 100).tolist()
# y_fit_3 = []
# for x in x_fit_1:
#     y_fit_3.append(fit_2(x, a, b, c))
#
# params, covarience = curve_fit(fit_2, extracted_features_switch[12], all_properties["InitialMassWeightedStellarAge"])
# a, b, c = params
# x_fit_4 = np.linspace(-20, 0, 100).tolist()
# y_fit_4 = []
# for x in x_fit_1:
#     y_fit_4.append(fit_2(x, a, b, c))
#
#
# fig, axs = plt.subplots(2, 2, figsize=(17, 10))
#
# axs[0, 0].scatter(x=extracted_features_switch[7], y=all_properties["StarFormationRate"], s=3, alpha=0.5)
# axs[0, 0].plot(x_fit_1, y_fit_1, c="black")
# axs[0, 0].set_xlabel("Feature 7", fontsize=20)
# axs[0, 0].set_ylabel("Star Formation Rate", fontsize=20)
# axs[0, 0].tick_params(labelsize=20)
#
# axs[0, 1].scatter(x=extracted_features_switch[12], y=all_properties["StarFormationRate"], s=3, alpha=0.5)
# axs[0, 1].plot(x_fit_2, y_fit_2, c="black")
# axs[0, 1].set_xlabel("Feature 12", fontsize=20)
# axs[0, 1].set_ylabel("Star Formation Rate", fontsize=20)
# axs[0, 1].tick_params(labelsize=20)
#
# axs[1, 0].scatter(x=extracted_features_switch[7], y=all_properties["re_r"], s=3, alpha=0.5)
# axs[1, 0].plot(x_fit_3, y_fit_3, c="black")
# axs[1, 0].set_xlabel("Feature 7", fontsize=20)
# axs[1, 0].set_ylabel("Semi-Major Axis", fontsize=20)
# axs[1, 0].tick_params(labelsize=20)
#
# axs[1, 1].scatter(x=extracted_features_switch[12], y=all_properties["InitialMassWeightedStellarAge"], s=3, alpha=0.5)
# axs[1, 1].plot(x_fit_4, y_fit_4, c="black")
# axs[1, 1].set_xlabel("Feature 12", fontsize=20)
# axs[1, 1].set_ylabel("Stellar Age", fontsize=20)
# axs[1, 1].tick_params(labelsize=20)
#
# plt.savefig("Plots/physical_scatter", bbox_inches='tight')
# plt.show()








# semi_major_7 = np.array([all_properties["re_r"]] + [extracted_features_switch[7]])
# semi_major_7 = semi_major_7.T
#
# # perform hierarchical ward clustering
# hierarchical = SpectralClustering(n_clusters=2)
#
# # get hierarchical clusters
# clusters = hierarchical.fit_predict(semi_major_7)
# all_properties["Semi-Major Type"] = clusters
#
#
# plt.figure(figsize=(20, 20))
#
# a = sns.scatterplot(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Semi-Major Type"], s=100)
# # plt.scatter(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Semi-Major Type"], s=3, alpha=0.5)
# # plt.plot(x_fit_3, y_fit_3, c="black")
# a.set_xlabel("Feature 7", fontsize=20)
# a.set_ylabel("Semi-Major Axis", fontsize=20)
# a.tick_params(labelsize=20)
#
# plt.show()



# spirals = all_properties["n_r"] <= 2.5
# all_properties["Spiral"] = spirals
#
# print(all_properties)
#
# plt.figure(figsize=(10, 8))
#
# # plt.scatter(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Spiral"])
# sns.scatterplot(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Spiral"])
# plt.show()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap



# encoding_dim = 24
#
# extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")



df1 = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")

structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)

print(df1["GalaxyID"].tolist())
print(df1)
print(structure_properties)

# plt.figure(figsize=(8,8))
#
# plt.scatter(y=structure_properties["re_r"], x=structure_properties["re_star"], s=1)
#
# plt.xlim(0, 120)
# plt.ylim(0, 120)
#
# plt.ylabel("re_r")
# plt.xlabel("re_star")
# plt.title("Semi-Major Axis")
#
# plt.savefig("Plots/semi-major_full")
# plt.show()




# structual features: Sersic index, Axis ratio, position angle, semi-major axis

# physical propeties: stellar mass, star formation rate, halo mass, black hole mass, merger history


relevant_feature_number = []
relevant_feature_ratio = []


for encoding_dim in range(17, 41):

    extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")

    extracted_features_switch = np.flipud(np.rot90(extracted_features))

    structure_correlation_df = pd.DataFrame(columns=["Sersic - r", "Sersic - star", "Axis Ratio - r", "Axis Ratio - star", "Semi-Major - r", "Semi-Major - star", "AB Magnitude"])

    for feature in range(0, len(extracted_features_switch)):

        # create a list to contain the correlation between that feature and each property
        correlation_list = []

        # loop through each property
        for gal_property in range(1, len(structure_properties.columns)):

            # calculate the correlation between that extracted feature and that property
            correlation = np.corrcoef(extracted_features_switch[feature], structure_properties.iloc[:, gal_property])[0][1]
            correlation_list.append(correlation)

        # add the correlation of that feature to the main dataframe
        structure_correlation_df.loc[len(structure_correlation_df)] = correlation_list

    # find the number of features at least slightly correlating with a property
    relevant_features = (abs(structure_correlation_df).max(axis=1) > 0.2).sum()


    relevant_feature_number.append(relevant_features)
    relevant_feature_ratio.append(relevant_features/encoding_dim)


plt.scatter(x=range(17, 41), y=relevant_feature_number)
plt.show()



# sns.histplot(data=structure_properties, x="q_r", bins=300)
# plt.show()







encoding_dim=29


extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")

extracted_features_switch = np.flipud(np.rot90(extracted_features))

structure_correlation_df = pd.DataFrame(columns=["Sersic - r", "Sersic - star", "Axis Ratio - r", "Axis Ratio - star", "Semi-Major - r", "Semi-Major - star", "AB Magnitude"])

for feature in range(0, len(extracted_features_switch)):

    # create a list to contain the correlation between that feature and each property
    correlation_list = []

    # loop through each property
    for gal_property in range(1, len(structure_properties.columns)):

        # calculate the correlation between that extracted feature and that property
        correlation = np.corrcoef(extracted_features_switch[feature], structure_properties.iloc[:, gal_property])[0][1]
        correlation_list.append(correlation)

    # add the correlation of that feature to the main dataframe
    structure_correlation_df.loc[len(structure_correlation_df)] = correlation_list



# set the figure size
plt.figure(figsize=(12, 16))

# sns.set(font_scale = 2)

# plot a heatmap for the dataframe (with annotations)
ax = sns.heatmap(abs(structure_correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})

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



plt.savefig("Plots/" + str(encoding_dim) + "_feature_property_correlation")
plt.show()

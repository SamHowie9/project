import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap




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
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# reset the index values
all_properties = all_properties.reset_index(drop=True)

# account for the testing dataset
all_properties = all_properties.iloc[:-200]

print(all_properties)








# fully balanced dataset

# for encoding_dim in range(5, 21):






encoding_dim = 30

correlation_df = pd.DataFrame(columns=["1e-3", "1e-4", "9e-5", "8e-5", "7e-5", "6e-5", "5e-5", "4e-5", "3e-5", "2e-5", "1e-5", "1e-6"])

for feature in range(0, encoding_dim):

    correlation_list = []

    for beta in ["001", "0001", "00009", "00008", "00007", "00006", "00005", "00004", "00003", "00002" "00001", "000001"]:


        # load the extracted features
        # extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
        extracted_features = np.load("Variational Eagle/Extracted Features/Test/bce_latent_" + str(encoding_dim) + "_beta_" + beta + ".npy")[0]
        encoding_dim = extracted_features.shape[1]
        extracted_features_switch = extracted_features.T

        print(extracted_features.shape)

        # remove augmented images
        extracted_features = extracted_features[:len(all_properties)]
        extracted_features_switch = extracted_features.T

        # perform pca on the extracted features
        # pca = PCA(n_components=11).fit(extracted_features)
        # extracted_features = pca.transform(extracted_features)
        # extracted_features = extracted_features[:len(all_properties)]
        # extracted_features_switch = extracted_features.T



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






fig, axs = plt.subplots(1, 1, figsize=(20, encoding_dim/2))

# sns.heatmap(abs(correlation_df), ax=axs, annot=False, cmap="Blues", cbar_kws={'label': 'Correlation'})


print(correlation_text_df)

sns.heatmap(abs(correlation_df), ax=axs, annot=correlation_text_df, fmt="", cmap="Blues", cbar_kws={'label': 'Correlation'})


# sns.heatmap(abs(correlation_df), ax=axs, annot=correlation_text_df, fmt="", cmap="Greys", alpha=0, cbar=False)


# axs.yticks(rotation=0)
# axs.set_tickparams(axis)
axs.set_yticks([])
axs.set_ylabel("Extracted Features", fontsize=15)
axs.xaxis.tick_top() # x axis on top
axs.xaxis.set_label_position('top')
axs.tick_params(length=0)
axs.figure.axes[-1].yaxis.label.set_size(15)

def wrap_labels(ax, width, break_long_words=False):

    labels = []
    # for label in ax.get_xticklabels():
        # text = label.get_text()

    label_names = ["β = 1e-3", "β = 1e-4", "β = 1e-5", "β = 1e-6"]

    for text in label_names:
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    axs.set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(axs, 10)


plt.savefig("Variational Eagle/Correlation Plots/Test/beta_sersic_sorted_2", bbox_inches='tight')
plt.show()






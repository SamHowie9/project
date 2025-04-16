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

correlation_df = pd.DataFrame(columns=["1e-3", "1e-4", "1e-5", "1e-6"])

for feature in range(0, encoding_dim):

    correlation_list = []

    for beta in ["001", "0001", "00001", "000001"]:


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





# order each of the columns (remove the number corresponding to each feature)
# correlation_df = pd.DataFrame({col: sorted(correlation_df[col], reverse=True) for col in correlation_df.columns})






plt.figure(figsize=(10, encoding_dim/2))

ax = sns.heatmap(abs(correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})

plt.yticks(rotation=0)
plt.ylabel("Extracted Features", fontsize=15)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(length=0)
ax.figure.axes[-1].yaxis.label.set_size(15)

def wrap_labels(ax, width, break_long_words=False):

    labels = []
    # for label in ax.get_xticklabels():
        # text = label.get_text()

    label_names = ["β = 1e-3", "β = 1e-4", "β = 1e-5", "β = 1e-6"]

    for text in label_names:
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(ax, 10)


plt.savefig("Variational Eagle/Correlation Plots/Test/beta_sersic", bbox_inches='tight')
plt.show()






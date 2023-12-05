import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import os

# load the extracted features
extracted_features = np.load("Features/8_features_new.npy")



# # load the two excel files into dataframes
# df1 = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
# df2 = pd.read_csv("stab3510_supplemental_file/table2.csv", comment="#")


df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])


# create the pairplot with custom marker size
kws = dict(s=10)
g = sns.pairplot(df, corner=True, plot_kws=kws)


# function to add the correlation coefficient to the plots
def corrfunc(x, y, ax=None, color=None):
    # find the correlation coefficient and round to 3 dp
    correlation = np.corrcoef(x, y)[0][1]
    correlation = np.round(correlation, decimals=3)

    # annotate the plot with the correlation coefficient
    ax = ax or plt.gca()
    ax.annotate(("ρ = " + str(correlation)), xy=(0.1, 1), xycoords=ax.transAxes)


# add the correlation coefficient to each of the scatter plots
g.map_lower(corrfunc)

# add some vertical space between the plots (given we are adding the correlation coefficients
plt.subplots_adjust(hspace=0.2)


plt.savefig("Plots/8_feature_histogram_new")
plt.show()

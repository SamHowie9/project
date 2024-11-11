from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


extracted_features = np.load("Variational Eagle/Extracted Features/Normalised to G/15_feature_300_epoch_features_1.npy")[0]

print(extracted_features.shape)

pca = PCA(n_components=0.90).fit(extracted_features.T)

print(pca.components_.shape)
import numpy as np
import pandas as pd



encoding_dim = 10

cae_features = np.load("Convolutional Eagle/Features Rand/" + str(encoding_dim) + "_features_1.npy")
vae_features = np.load("Variational Eagle/Extracted Features/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")

# print(cae_features)

print(cae_features.shape)
print(vae_features[0].shape)

# print(features.shape)
#
# print(features)
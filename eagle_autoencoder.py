import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# select to use GPU 0 on cosma
os.environ["CUDA_VISIBLE_DEVICES"]="2" # for GPU




# stores an empty list to contain all the image data to train the model
all_images = []

# load the supplemental file into a dataframe
df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")

# loop through each galaxy in the supplmental file
for i, galaxy in enumerate(df["GalaxyID"].tolist()):

    # get the filename for that galaxy
    filename = "galface_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    all_images.append(image)





# split the data into training and testing data based on this number (and convert from list to numpy array of shape (256,256,3) given it is an rgb image
train_images = np.array(all_images[:-200])
test_images = np.array(all_images[-200:])
# train_images = np.array(all_images)





# set the encoding dimension (number of extracted features)
encoding_dim = 16



# Define keras tensor for the encoder
input_image = keras.Input(shape=(256, 256, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 128, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (8, 8, 4)
x = Flatten()(x)                                                                                    # (256)
x = Dense(units=32)(x)                                                                              # (32)
encoded = Dense(units=encoding_dim, name="encoded")(x)                                              # (2)


# layers for the decoder
x = Dense(units=32)(encoded)                                                                        # (32)
x = Dense(units=256)(x)                                                                             # (256)
x = Reshape((8, 8, 4))(x)                                                                           # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (128, 128, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)        # (256, 256, 3)



# crate autoencoder
autoencoder = keras.Model(input_image, decoded)



# crate autoencoder
autoencoder = keras.Model(input_image, decoded)

# create the encoder using the autoencoder layers
encoder = keras.Sequential()
for i in range(0, 9):
    encoder.add(autoencoder.layers[i])

# create the decoder using the autoencoder layers
decoder = keras.Sequential()
for i in range(9, 18):
    decoder.add(autoencoder.layers[i])

# build the decoder
decoder.build(input_shape=(None, encoding_dim))


# root means squared loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss=root_mean_squared_error)


# train the model
model_data = autoencoder.fit(train_images, train_images, epochs=200, batch_size=1)

# save the weights
autoencoder.save_weights(filepath="Weights/" + str(encoding_dim) + "_feature_weights.h5", overwrite=True)


# load the weights
# autoencoder.load_weights("Weights/" + str(encoding_dim) + "_feature_weights.h5")


# extract the features
extracted_features = encoder.predict(train_images)

# save the features as a numpy array
np.save("Features/" + str(encoding_dim) + "_features.npy", extracted_features)






# create a subset of the validation data to reconstruct (first 10 images)
images_to_reconstruct = test_images[:10]

# number of images to reconstruct
n = 10

# reconstruct the images
reconstructed_images = autoencoder.predict(test_images[:n])

# create figure to hold subplots
fig, axs = plt.subplots(3, n-1, figsize=(20,5))

# plot each subplot
for i in range(0, n-1):

    # show the original image (remove axes)
    axs[0,i].imshow(test_images[i])
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    # show the reconstructed image (remove axes)
    axs[1,i].imshow(reconstructed_images[i])
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)

    # calculate residue (difference between two images) and show this
    residue_image = np.absolute(np.subtract(reconstructed_images[i], test_images[i]))
    axs[2,i].imshow(residue_image)
    axs[2,i].get_xaxis().set_visible(False)
    axs[2,i].get_yaxis().set_visible(False)

plt.savefig("Plots/" + str(encoding_dim) + "_feature_reconstruction")
plt.show()



# plot the training and validation loss
plt.plot(model_data.history["loss"], label="training data")
plt.plot(model_data.history["val_loss"], label="validation data")
plt.legend()

plt.savefig("Plots/" + str(encoding_dim) + "_feature_loss")
plt.show()

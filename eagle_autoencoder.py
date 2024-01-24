from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
from keras import backend as K
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'




# select which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img



def half_max_range(image):

    mean_intensity = image.mean()

    intensity_x = image.mean(axis=2).mean(axis=0)
    intensity_y = image.mean(axis=2).mean(axis=1)

    half_max_intensity_x = np.max(intensity_x/mean_intensity) / 2
    half_max_intensity_y = np.max(intensity_y/mean_intensity) / 2


    size = len(intensity_x)

    start_x = 0
    start_y = 0
    end_x = 255
    end_y = 255

    found_start_x = False
    found_start_y = False
    found_end_x = False
    found_end_y = False

    # loop through half of the image
    for j in range(0, int(size / 2)):


        # if we haven't previously found the cutoff point and are still below the cutoff, increment the pointer
        if (found_start_x is False) and ((intensity_x[j] / mean_intensity) < half_max_intensity_x):
            start_x += 1
        else:
            found_start_x = True

        if (found_end_x is False) and ((intensity_x[-j] / mean_intensity) < half_max_intensity_x):
            end_x -= 1
        else:
            found_end_x = True

        if (found_start_y is False) and ((intensity_y[j] / mean_intensity) < half_max_intensity_y):
            start_y += 1
        else:
            found_start_y = True

        if (found_end_y is False) and ((intensity_y[-j] / mean_intensity) < half_max_intensity_y):
            end_y -= 1
        else:
            found_end_y = True

    return start_x, end_x, start_y, end_y



def resize_image(image, cutoff=60):

    # get the fill width half maximum (for x and y direction)
    start_x, end_x, start_y, end_y = half_max_range(image)

    # calculate the full width half maximum
    range_x = end_x - start_x
    range_y = end_y - start_y

    # check if the majority of out image is within the cutoff range, if so, center crop, otherwise, scale image down
    if (range_x <= cutoff) and (range_y <= cutoff):
        image = center_crop(image, (128, 128))
    else:
        image = cv2.resize(image, (128, 128))

    # return the resized image
    return image





# stores an empty list to contain all the image data to train the model
all_images = []

# load the supplemental file into a dataframe
df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")

# loop through each galaxy in the supplemental file
for i, galaxy in enumerate(df["GalaxyID"].tolist()):

    # get the filename for that galaxy
    filename = "galface_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    image = resize_image(image=image)
    all_images.append(image)





# split the data into training and testing data based on this number (and convert from list to numpy array of shape (256,256,3) given it is an rgb image
train_images = np.array(all_images[:-200])
test_images = np.array(all_images[-200:])
# train_images = np.array(all_images)





# set the encoding dimension (number of extracted features)
encoding_dim = 20


# Define keras tensor for the encoder
input_image = keras.Input(shape=(128, 128, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 128, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (8, 8, 4)
x = Flatten()(x)                                                                                    # (256)
x = Dense(units=64)(x)                                                                              # (32)
encoded = Dense(units=encoding_dim, name="encoded")(x)                                              # (2)


# layers for the decoder
x = Dense(units=64)(encoded)                                                                        # (32)
x = Dense(units=256)(x)                                                                             # (256)
x = Reshape((8, 8, 4))(x)                                                                           # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (128, 128, 32)
# x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)        # (256, 256, 3)



# crate autoencoder
autoencoder = keras.Model(input_image, decoded)
autoencoder.summary()



# create the encoder using the autoencoder layers
encoder = keras.Sequential()
for i in range(0, 9):
    encoder.add(autoencoder.layers[i])
encoder.summary()

# create the decoder using the autoencoder layers
decoder = keras.Sequential()
for i in range(9, 16):
    decoder.add(autoencoder.layers[i])

# build the decoder
decoder.build(input_shape=(None, encoding_dim))
decoder.summary()




# root means squared loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss=root_mean_squared_error)




# train the model
model_data = autoencoder.fit(train_images, train_images, epochs=300, batch_size=1, validation_data=(test_images, test_images))

# load the weights
# autoencoder.load_weights("Weights/" + str(encoding_dim) + "_feature_weights.h5")

# save the weights
autoencoder.save_weights(filepath="Weights/" + str(encoding_dim) + "_feature_weights.h5", overwrite=True)





# extract the features
extracted_features = encoder.predict(train_images)

# save the features as a numpy array
np.save("Features/" + str(encoding_dim) + "_features.npy", extracted_features)



# extracted_features_switch = np.flipud(np.rot90(extracted_features))
#
# mean_features = []
#
# for i in range(encoding_dim):
#     mean_features[i] = median(extracted_features_switch[i])
#
# print(mean_features)
#
# latent_features = []
#
# for i in range(encoding_dim):






# # create a subset of the validation data to reconstruct (first 10 images)
# images_to_reconstruct = test_images[:10]
#
# # number of images to reconstruct
# n = 10
#
# # reconstruct the images
# reconstructed_images = autoencoder.predict(test_images[:n])
#
# # create figure to hold subplots
# fig, axs = plt.subplots(3, n-1, figsize=(20,5))
#
# # plot each subplot
# for i in range(0, n-1):
#
#     # show the original image (remove axes)
#     axs[0,i].imshow(test_images[i])
#     axs[0,i].get_xaxis().set_visible(False)
#     axs[0,i].get_yaxis().set_visible(False)
#
#     # show the reconstructed image (remove axes)
#     axs[1,i].imshow(reconstructed_images[i])
#     axs[1,i].get_xaxis().set_visible(False)
#     axs[1,i].get_yaxis().set_visible(False)
#
#     # calculate residue (difference between two images) and show this
#     residue_image = np.absolute(np.subtract(reconstructed_images[i], test_images[i]))
#     axs[2,i].imshow(residue_image)
#     axs[2,i].get_xaxis().set_visible(False)
#     axs[2,i].get_yaxis().set_visible(False)
#
# plt.savefig("Plots/" + str(encoding_dim) + "_feature_reconstruction_2")
# plt.show()






print(model_data.history["loss"][-1], model_data.history["val_loss"][-1])
loss = np.array([model_data.history["loss"][-1], model_data.history["val_loss"][-1]])
print()
print(encoding_dim)
print(loss)
np.save("Loss/" + str(encoding_dim) + "_feature_loss", loss)





# # plot the training and validation loss
# plt.plot(model_data.history["loss"], label="training data")
# plt.plot(model_data.history["val_loss"], label="validation data")
# plt.legend()
#
# plt.savefig("Plots/" + str(encoding_dim) + "_feature_loss")
# plt.show()

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
train_images = []

# loop through the directory containing all the image files
for file in os.listdir("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/"):

    # open the fits file and get the image data (this is a numpy array of each pixel value)
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/" + file)

    # append the image data to the main list containing all data of all the images
    train_images.append(image)

# # return this list
# return train_images


# get the images and labels to train the model
# train_images = get_images()


# find the number of images that you will test the model on
testing_count = int(len(train_images)/10)

# split the data into training and testing data based on this number (and convert from list to numpy array of shape (256,256,3) given it is an rgb image
train_images = np.array(train_images[:testing_count*10])
test_images = np.array(train_images[testing_count:])


# # Instantiate a Keras tensor to allow us to build the model
# input_image = keras.Input(shape=(256, 256, 3))
#
# # layers for the encoder
# encoded = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(input_image)
#
# # layers for the decoder (extra one with 1 filter to get back to the correct shape)
# x = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(encoded)
# decoded = Conv2D(filters=3, kernel_size=3, activation="sigmoid", padding="same")(x)



# # Define keras tensor for the encoder
# input_image = keras.Input(shape=(256, 256, 3))                                                      # (256, 256, 3)
#
# # layers for the encoder
# x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 32)
# x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                      # (64, 64, 16)
# x = Flatten()(x)                                                                                            # (65536) = (64 * 64 * 16)
# # x = Dense(units=16384)(x)
# x = Dense(units=1028)(x)
# # x = Dense(units=256)(x)
# # x = Dense(units=32)(x)
# encoded = Dense(units=32, activation="relu", name="z_mean")(x)                                                # (2)
#
# # build the encoder
# # encoder = keras.Model(inputs=input_image, outputs=encoded, name="encoder")
#
# # # Define keras tensor for the decoder
# # input_image_decoder = keras.Input(shape=(2))                                                                # (2)
#
# # layers for the decoder
# # x = Dense(units=32)(encoded)
# # x = Dense(units=256)(x)
# # x = Dense(units=1028)(encoded)
# x = Dense(units=16384)(encoded)
# # x = Dense(units=64*64*32, activation="relu")(x)                                           # (131072) = (64 * 64 * 32)
# x = Dense(units=64*64*16, activation="relu")(x)                                           # (65536) = (64 * 64 * 16)
# x = Reshape((64, 64, 16))(x)                                                                                # (64, 64, 16)
# x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (128, 128, 64)
# x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (265, 256, 32)
# decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same")(x)                   # (256, 256, 3)


# set the encoding dimension (number of extracted features)
encoding_dim = 8

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
encoded = Dense(units=encoding_dim, name="encoded")(x)                                                         # (2)


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

encoder = keras.Sequential()
for i in range(0, 9):
    encoder.add(autoencoder.layers[i])

decoder = keras.Sequential()
for i in range(9, 18):
    decoder.add(autoencoder.layers[i])

decoder.build(input_shape=(None, encoding_dim))





# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# train the model
model_data = autoencoder.fit(train_images, train_images, epochs=150, batch_size=1, validation_data=(test_images, test_images))

# save the weights
autoencoder.save_weights(filepath="Weights/8_feature_weights.h5", overwrite=True)


# load the weights
# autoencoder.load_weights("Weights/7_feature_weights.h5")




# # extract encoder layer and decoder layer from autoencoder
# encoder_layer = autoencoder.get_layer("encoded")
# decoder_layer = autoencoder.get_layer("decoded")
#
# # get the shape of the decoder input
# decoder_input = keras.Input(shape=encoding_dim)
#
# # build the encoder
# encoder = keras.Model(autoencoder.input, encoder_layer.output)
#
# # build the decoder
# decoder = keras.Model(decoder_input, autoencoder.layers[-1](decoder_input))



# extract the features
extracted_features = encoder.predict(train_images)

# save the features as a numpy array
np.save("Features/8_features.npy", extracted_features)





# # define encoded input
# encoded_input = keras.Input(shape=(2))
#
#
# # crete encoder
# encoder = keras.Model(autoencoder.input, encoder_layer.output)
#
# # create decoder
# decoder = keras.Model(encoder_layer.output, decoder_layer.output)
# decoder.summary()





# autoencoder.load_weights("Weights/8_feature_weights.h5")


# # number of galaxies on each side
# n = 15
#
# # size of each image
# image_size = 256
#
# # create the figure to store the images
# figure = np.zeros((image_size * n, image_size * n))
#
# # sample points within [-15, 15] standard deviations
# grid_x = np.linspace(-8, 8, n)
# grid_y = np.linspace(-8, 8, n)
#
# # populate each point on the figure
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#
#         z_sample = np.array([[xi, yi]])
#
#         x_decoded = decoder.predict(z_sample)
#
#         image = x_decoded[0].reshape(image_size, image_size, 3)
#
#         # add image to the figure
#         figure[i * image_size: (i+1) * image_size,
#                j * image_size: (j+1) * image_size] = image
#
# plt.figure(figsize=(20, 20))
# plt.imshow(figure)



# create a subset of the validation data to reconstruct (first 10 images)
images_to_reconstruct = test_images[:10]

# number of images to reconstruct
n = 10

# reconstruct the images
reconstructed_images = autoencoder.predict(test_images[:n])

# create figure to hold subplots
# fig, axs = plt.subplots(4, n-1, figsize=(20,8))
fig, axs = plt.subplots(2, n-1, figsize=(20,4))

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

    # # calculate residue (difference between two images) and show this
    # residue_image = np.absolute(np.subtract(reconstructed_images[i], test_images[i]))
    # axs[2,i].imshow(residue_image)
    # axs[2,i].get_xaxis().set_visible(False)
    # axs[2,i].get_yaxis().set_visible(False)
    #
    # # add an exponential transform to the residue to show differences more clearly
    # exponential_residue = np.exp(5 * residue_image) - 1
    # axs[3,i].imshow(exponential_residue)
    # axs[3,i].get_xaxis().set_visible(False)
    # axs[3,i].get_yaxis().set_visible(False)



# # build the encoder for feature extraction
# encoder = keras.Model(input_image, encoded)
# extracted_features = encoder.predict(train_images)
#
# np.save("Features/8_features", extracted_features)

# print(extracted_features)
# print()
# print()
# print(np.array2string(extracted_features))
#
# # write the extracted features to a file to save them
# f = open("Features/7_features", "w")
# f.write(np.array2string(extracted_features))
# f.close()

# print(extracted_features.tolist())
#
# # lists to store the values of each image for each extracted feature
# f1 = []
# f2 = []
# f3 = []
# f4 = []
# f5 = []
# f6 = []
# f7 = []
#
# # loop through each pair of values for each image and add the values to the individual lists
# for i in range(extracted_features.shape[0]):
#     f1.append(extracted_features[i][0])
#     f2.append(extracted_features[i][1])
#     f3.append(extracted_features[i][2])
#     f4.append(extracted_features[i][3])
#     f5.append(extracted_features[i][4])
#     f6.append(extracted_features[i][5])
#     f7.append(extracted_features[i][6])
#
#
#
# # linear regression via least squares between each of the 3 features
# # b_21, a_21 = np.polyfit(f2, f1, deg=1)
# # b_31, a_31 = np.polyfit(f3, f1, deg=1)
# # b_41, a_41 = np.polyfit(f4, f1, deg=1)
# # b_51, a_51 = np.polyfit(f5, f1, deg=1)
# # b_32, a_32 = np.polyfit(f3, f2, deg=1)
# # b_42, a_42 = np.polyfit(f4, f2, deg=1)
# # b_43, a_43 = np.polyfit(f4, f3, deg=1)
# # b_52, a_52 = np.polyfit(f5, f2, deg=1)
# # b_53, a_53 = np.polyfit(f5, f3, deg=1)
# # b_54, a_54 = np.polyfit(f5, f4, deg=1)
# # b_21, a_21 = np.polyfit(f2, f1, deg=1)
# # b_31, a_31 = np.polyfit(f3, f1, deg=1)
# # b_41, a_41 = np.polyfit(f4, f1, deg=1)
# # b_51, a_51 = np.polyfit(f5, f1, deg=1)
# # b_61, a_61 = np.polyfit(f6, f1, deg=1)
# # b_32, a_32 = np.polyfit(f3, f2, deg=1)
# # b_42, a_42 = np.polyfit(f4, f2, deg=1)
# # b_52, a_52 = np.polyfit(f5, f2, deg=1)
# # b_62, a_62 = np.polyfit(f6, f2, deg=1)
# # b_43, a_43 = np.polyfit(f4, f3, deg=1)
# # b_53, a_53 = np.polyfit(f5, f3, deg=1)
# # b_63, a_63 = np.polyfit(f6, f3, deg=1)
# # b_54, a_54 = np.polyfit(f5, f4, deg=1)
# # b_64, a_64 = np.polyfit(f6, f4, deg=1)
# # b_65, a_65 = np.polyfit(f6, f5, deg=1)
# b_21, a_21 = np.polyfit(f2, f1, deg=1)
# b_31, a_31 = np.polyfit(f3, f1, deg=1)
# b_41, a_41 = np.polyfit(f4, f1, deg=1)
# b_51, a_51 = np.polyfit(f5, f1, deg=1)
# b_61, a_61 = np.polyfit(f6, f1, deg=1)
# b_71, a_71 = np.polyfit(f7, f1, deg=1)
#
# b_32, a_32 = np.polyfit(f3, f2, deg=1)
# b_42, a_42 = np.polyfit(f4, f2, deg=1)
# b_52, a_52 = np.polyfit(f5, f2, deg=1)
# b_62, a_62 = np.polyfit(f6, f2, deg=1)
# b_72, a_72 = np.polyfit(f7, f2, deg=1)
#
# b_43, a_43 = np.polyfit(f4, f3, deg=1)
# b_53, a_53 = np.polyfit(f5, f3, deg=1)
# b_63, a_63 = np.polyfit(f6, f3, deg=1)
# b_73, a_73 = np.polyfit(f7, f3, deg=1)
#
# b_54, a_54 = np.polyfit(f5, f4, deg=1)
# b_64, a_64 = np.polyfit(f6, f4, deg=1)
# b_74, a_74 = np.polyfit(f7, f4, deg=1)
#
# b_65, a_65 = np.polyfit(f6, f5, deg=1)
# b_75, a_75 = np.polyfit(f7, f5, deg=1)
#
# b_76, a_76 = np.polyfit(f7, f6, deg=1)
#
#
#
# # Create sequence of 100 numbers from the minimum feature 1 value to the maximum feature 1 value (for regression line)
# sequence_f2 = np.linspace(np.min(f2), np.max(f2), num=100)
# sequence_f3 = np.linspace(np.min(f3), np.max(f3), num=100)
# sequence_f4 = np.linspace(np.min(f4), np.max(f4), num=100)
# sequence_f5 = np.linspace(np.min(f5), np.max(f5), num=100)
# sequence_f6 = np.linspace(np.min(f6), np.max(f6), num=100)
# sequence_f7 = np.linspace(np.min(f7), np.max(f7), num=100)
#
#
#
# # create the figure for the plot
# fig, axs = plt.subplots(4, 7, figsize=(40, 20))
#
# # plot feature 1
# axs[0][0].hist(f1, bins=40)
# axs[0][0].set_title("Feature 1")
#
# # plot feature 2
# axs[0][1].hist(f2, bins=40)
# axs[0][1].set_title("Feature 2")
#
# # plot feature 3
# axs[0][2].hist(f3, bins=40)
# axs[0][2].set_title("Feature 3")
#
# # plot feature 4
# axs[0][3].hist(f4, bins=40)
# axs[0][3].set_title("Feature 4")
#
# # plot feature 5
# axs[0][4].hist(f5, bins=40)
# axs[0][4].set_title("Feature 5")
#
# # plot feature 6
# axs[0][5].hist(f6, bins=40)
# axs[0][5].set_title("Feature 6")
#
# # plot feature 7
# axs[0][6].hist(f6, bins=40)
# axs[0][6].set_title("Feature 7")
#
#
# # correlation between 2 and 1
# axs[1][0].scatter(f2, f1, s=5)
# axs[1][0].plot(sequence_f2, a_21 + b_21 * sequence_f2, color="k", lw=1.5)
# axs[1][0].set_title("Feature 1 Against Feature 2")
# axs[1][0].set_xlabel("Feature 2")
# axs[1][0].set_ylabel("Feature 1")
#
# # correlation between 3 and 1
# axs[1][1].scatter(f3, f1, s=5)
# axs[1][1].plot(sequence_f3, a_31 + b_31 * sequence_f3, color="k", lw=1.5)
# axs[1][1].set_title("Feature 1 Against Feature 3")
# axs[1][1].set_xlabel("Feature 3")
# axs[1][1].set_ylabel("Feature 1")
#
# # correlation between 4 and 1
# axs[1][2].scatter(f4, f1, s=5)
# axs[1][2].plot(sequence_f4, a_41 + b_41 * sequence_f4, color="k", lw=1.5)
# axs[1][2].set_title("Feature 1 Against Feature 4")
# axs[1][2].set_xlabel("Feature 4")
# axs[1][2].set_ylabel("Feature 1")
#
# # correlation between 5 and 1
# axs[1][3].scatter(f5, f1, s=5)
# axs[1][3].plot(sequence_f5, a_51 + b_51 * sequence_f5, color="k", lw=1.5)
# axs[1][3].set_title("Feature 1 Against Feature 5")
# axs[1][3].set_xlabel("Feature 5")
# axs[1][3].set_ylabel("Feature 1")
#
# # correlation between 6 and 1
# axs[1][4].scatter(f6, f1, s=5)
# axs[1][4].plot(sequence_f6, a_61 + b_61 * sequence_f6, color="k", lw=1.5)
# axs[1][4].set_title("Feature 1 Against Feature 6")
# axs[1][4].set_xlabel("Feature 6")
# axs[1][4].set_ylabel("Feature 1")
#
# # correlation between 7 and 1
# axs[1][5].scatter(f7, f1, s=5)
# axs[1][5].plot(sequence_f7, a_71 + b_71 * sequence_f7, color="k", lw=1.5)
# axs[1][5].set_title("Feature 1 Against Feature 7")
# axs[1][5].set_xlabel("Feature 7")
# axs[1][5].set_ylabel("Feature 1")
#
#
# # correlation between 3 and 2
# axs[1][6].scatter(f3, f2, s=5)
# axs[1][6].plot(sequence_f3, a_32 + b_32 * sequence_f3, color="k", lw=1.5)
# axs[1][6].set_title("Feature 2 Against Feature 3")
# axs[1][6].set_xlabel("Feature 3")
# axs[1][6].set_ylabel("Feature 2")
#
# # correlation between 4 and 2
# axs[2][0].scatter(f4, f2, s=5)
# axs[2][0].plot(sequence_f4, a_42 + b_42 * sequence_f4, color="k", lw=1.5)
# axs[2][0].set_title("Feature 2 Against Feature 4")
# axs[2][0].set_xlabel("Feature 4")
# axs[2][0].set_ylabel("Feature 2")
#
# # correlation between 5 and 2
# axs[2][1].scatter(f5, f2, s=5)
# axs[2][1].plot(sequence_f5, a_52 + b_52 * sequence_f5, color="k", lw=1.5)
# axs[2][1].set_title("Feature 2 Against Feature 5")
# axs[2][1].set_xlabel("Feature 5")
# axs[2][1].set_ylabel("Feature 2")
#
# # correlation between 6 and 2
# axs[2][2].scatter(f6, f2, s=5)
# axs[2][2].plot(sequence_f6, a_62 + b_62 * sequence_f6, color="k", lw=1.5)
# axs[2][2].set_title("Feature 2 Against Feature 6")
# axs[2][2].set_xlabel("Feature 6")
# axs[2][2].set_ylabel("Feature 2")
#
# # correlation between 7 and 2
# axs[2][3].scatter(f7, f2, s=5)
# axs[2][3].plot(sequence_f7, a_72 + b_72 * sequence_f7, color="k", lw=1.5)
# axs[2][3].set_title("Feature 2 Against Feature 7")
# axs[2][3].set_xlabel("Feature 7")
# axs[2][3].set_ylabel("Feature 2")
#
#
# # correlation between 4 and 3
# axs[2][4].scatter(f4, f3, s=5)
# axs[2][4].plot(sequence_f4, a_43 + b_43 * sequence_f4, color="k", lw=1.5)
# axs[2][4].set_title("Feature 3 Against Feature 4")
# axs[2][4].set_xlabel("Feature 4")
# axs[2][4].set_ylabel("Feature 3")
#
# # correlation between 5 and 3
# axs[2][5].scatter(f5, f3, s=5)
# axs[2][5].plot(sequence_f5, a_53 + b_53 * sequence_f5, color="k", lw=1.5)
# axs[2][5].set_title("Feature 3 Against Feature 5")
# axs[2][5].set_xlabel("Feature 5")
# axs[2][5].set_ylabel("Feature 3")
#
# # correlation between 6 and 3
# axs[2][6].scatter(f6, f3, s=5)
# axs[2][6].plot(sequence_f6, a_63 + b_63 * sequence_f6, color="k", lw=1.5)
# axs[2][6].set_title("Feature 3 Against Feature 6")
# axs[2][6].set_xlabel("Feature 6")
# axs[2][6].set_ylabel("Feature 3")
#
# # correlation between 7 and 3
# axs[3][0].scatter(f7, f3, s=5)
# axs[3][0].plot(sequence_f7, a_73 + b_73 * sequence_f7, color="k", lw=1.5)
# axs[3][0].set_title("Feature 3 Against Feature 7")
# axs[3][0].set_xlabel("Feature 7")
# axs[3][0].set_ylabel("Feature 3")
#
#
# # correlation between 5 and 4
# axs[3][1].scatter(f5, f4, s=5)
# axs[3][1].plot(sequence_f5, a_54 + b_54 * sequence_f5, color="k", lw=1.5)
# axs[3][1].set_title("Feature 4 Against Feature 5")
# axs[3][1].set_xlabel("Feature 5")
# axs[3][1].set_ylabel("Feature 4")
#
# # correlation between 6 and 4
# axs[3][2].scatter(f6, f4, s=5)
# axs[3][2].plot(sequence_f6, a_64 + b_64 * sequence_f6, color="k", lw=1.5)
# axs[3][2].set_title("Feature 4 Against Feature 6")
# axs[3][2].set_xlabel("Feature 6")
# axs[3][2].set_ylabel("Feature 4")
#
# # correlation between 7 and 4
# axs[3][3].scatter(f7, f4, s=5)
# axs[3][3].plot(sequence_f7, a_74 + b_74 * sequence_f7, color="k", lw=1.5)
# axs[3][3].set_title("Feature 4 Against Feature 7")
# axs[3][3].set_xlabel("Feature 7")
# axs[3][3].set_ylabel("Feature 4")
#
#
# # correlation between 6 and 5
# axs[3][4].scatter(f6, f5, s=5)
# axs[3][4].plot(sequence_f6, a_65 + b_65 * sequence_f6, color="k", lw=1.5)
# axs[3][4].set_title("Feature 5 Against Feature 6")
# axs[3][4].set_xlabel("Feature 6")
# axs[3][4].set_ylabel("Feature 5")
#
# # correlation between 7 and 5
# axs[3][5].scatter(f7, f5, s=5)
# axs[3][5].plot(sequence_f7, a_75 + b_75 * sequence_f7, color="k", lw=1.5)
# axs[3][5].set_title("Feature 5 Against Feature 7")
# axs[3][5].set_xlabel("Feature 7")
# axs[3][5].set_ylabel("Feature 5")
#
# # correlation between 7 and 6
# axs[3][6].scatter(f7, f6, s=5)
# axs[3][6].plot(sequence_f7, a_76 + b_76 * sequence_f7, color="k", lw=1.5)
# axs[3][6].set_title("Feature 6 Against Feature 7")
# axs[3][6].set_xlabel("Feature 7")
# axs[3][6].set_ylabel("Feature 6")
#
#
# # axs[3][3].set_axis_off()
# # axs[3][4].set_axis_off()
# # axs[3][5].set_axis_off()











# plt.plot(model_data.history["loss"], label="training data")
# plt.plot(model_data.history["val_loss"], label="validation data")
# plt.legend()



plt.savefig("Plots/8_feature_reconstruction")
plt.show()

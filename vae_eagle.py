import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg





encoding_dim = 25
run = 2

# select which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# number of epochs for run
epochs = 200

# batch size for run
batch_size = 32





# for run in range(2, 4):
for run in [2]:

    # normalise each band individually
    def normalise_independently(image):
        image = image.T
        for i in range(0, 3):
            image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
        return image.T

    # normalise each band to r
    def normalise_to_r(image):
        image = image.T
        for i in range(0, 3):
            image[i] = (image[i] - np.min(image[i])) / (np.max(image[1]) - np.min(image[1]))
        return image.T






    # load the original dataset

    # # load structural and physical properties into dataframes
    # structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
    # physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
    #
    # # dataframe for all properties
    # all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
    #
    # # find all bad fit galaxies
    # bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
    # print("Bad Fit Indices:", bad_fit)
    #
    # # remove those galaxies
    # for galaxy in bad_fit:
    #     all_properties = all_properties.drop(galaxy, axis=0)
    #
    # # get a list of all the ids of the galaxies
    # chosen_galaxies = list(all_properties["GalaxyID"])
    #
    # # list to contain all galaxy images
    # all_images = []
    #
    # # # loop through each galaxy in the supplemental file
    # for i, galaxy in enumerate(chosen_galaxies):
    #
    #     # get the filename of each galaxy in the supplemental file
    #     filename = "galrand_" + str(galaxy) + ".png"
    #
    #     # open the image and append it to the main list
    #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    #
    #     # find smallest non zero pixel value in the image and replace all zero values with this (for log transformation)
    #     smallest_non_zero = np.min(image[image > 0])
    #     image = np.where(image == 0.0, smallest_non_zero, image)
    #
    #     # normalise the image (either each band independently or to the r band)
    #     image = normalise_independently(image)
    #
    #     # add the image to the dataset
    #     all_images.append(image)
    #
    #
    # # split the data into training and testing data (200 images used for testing)
    # train_images = np.array(all_images[:-200])
    # test_images = np.array(all_images[-200:])









    # load the images as a fully balanced dataset

    # load structural and physical properties into dataframes
    structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
    physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

    # dataframe for all properties
    all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

    # find all bad fit galaxies
    bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
    print("Bad Fit Indices:", bad_fit)

    # remove those galaxies
    for galaxy in bad_fit:
        all_properties = all_properties.drop(galaxy, axis=0)

    # get a list of all the ids of the galaxies
    chosen_galaxies = list(all_properties["GalaxyID"])

    # list to contain all galaxy images
    all_images = []

    # # loop through each galaxy
    for i, galaxy in enumerate(chosen_galaxies):

        # get the filename of each galaxy in the supplemental file
        filename = "galrand_" + str(galaxy) + ".png"

        # open the image and append it to the main list
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

        # normalise the image (each band independently)
        image = normalise_independently(image)

        # add the image to the dataset
        all_images.append(image)


    # split the data into training and testing data (200 images used for testing)
    train_images = all_images[:-200]
    test_images = np.array(all_images[-200:])


    # load the filenames of the augmented elliptical images
    augmented_galaxies =  os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/")

    for galaxy in augmented_galaxies:

        # load each augmented image
        image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/" + galaxy)

        # normalise the image
        image = normalise_independently(image)

        # add the image to the training set (not the testing set)
        train_images.append(image)



    # load the filenames of the augmented unknown images
    augmented_galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/")

    print("...")

    for galaxy in augmented_galaxies:

        # load each augmented image
        image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/" + galaxy)

        # normalise the image
        image = normalise_independently(image)

        # add the image to the training set (not the testing set)
        train_images.append(image)

    # convert the training set to a numpy array
    train_images = np.array(train_images)


    print(train_images.shape)
    print(test_images.shape)
    print()







    # load only the spiral galaxies

    # # load structural and physical properties into dataframes
    # structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
    # physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
    #
    # # dataframe for all properties
    # all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
    #
    # # find all bad fit galaxies
    # bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
    # print("Bad Fit Indices:", bad_fit)
    #
    # # remove those galaxies
    # for galaxy in bad_fit:
    #     all_properties = all_properties.drop(galaxy, axis=0)
    #
    #
    # # take only the sprial galaxies
    # all_properties = all_properties[all_properties["n_r"] <= 2.5]
    #
    # # get a list of all the ids of the galaxies
    # chosen_galaxies = list(all_properties["GalaxyID"])
    #
    # # list to contain all galaxy images
    # all_images = []
    #
    # # # loop through each galaxy
    # for i, galaxy in enumerate(chosen_galaxies):
    #
    #     # get the filename of each galaxy in the supplemental file
    #     filename = "galrand_" + str(galaxy) + ".png"
    #
    #     # open the image and append it to the main list
    #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    #
    #     # normalise the image (each band independently)
    #     image = normalise_independently(image)
    #
    #     # add the image to the dataset
    #     all_images.append(image)
    #
    #
    # # split the data into training and testing data (200 images used for testing)
    # train_images = np.array(all_images[:-200])
    # test_images = np.array(all_images[-200:])
    #
    # print(train_images.shape)
    # print(test_images.shape)
    # print()






    # load only the 'unknown' galaxies

    # # load structural and physical properties into dataframes
    # structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
    # physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
    #
    # # dataframe for all properties
    # all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
    #
    # # find all bad fit galaxies
    # bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
    # print("Bad Fit Indices:", bad_fit)
    #
    # # remove those galaxies
    # for galaxy in bad_fit:
    #     all_properties = all_properties.drop(galaxy, axis=0)
    #
    #
    # # take only the sprial galaxies
    # all_properties = all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")]
    #
    # # get a list of all the ids of the galaxies
    # chosen_galaxies = list(all_properties["GalaxyID"])
    #
    # # list to contain all galaxy images
    # all_images = []
    #
    # # # loop through each galaxy
    # for i, galaxy in enumerate(chosen_galaxies):
    #
    #     # get the filename of each galaxy in the supplemental file
    #     filename = "galrand_" + str(galaxy) + ".png"
    #
    #     # open the image and append it to the main list
    #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    #
    #     # normalise the image (each band independently)
    #     image = normalise_independently(image)
    #
    #     # add the image to the dataset
    #     all_images.append(image)
    #
    #
    # # split the data into training and testing data (12 images used for testing)
    # train_images = np.array(all_images[:-12])
    # test_images = np.array(all_images[-12:])
    #
    # print(train_images.shape)
    # print(test_images.shape)
    # print()










    # load only the elliptical galaxies

    # # load structural and physical properties into dataframes
    # structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
    # physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
    #
    # # dataframe for all properties
    # all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
    #
    # # find all bad fit galaxies
    # bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
    # print("Bad Fit Indices:", bad_fit)
    #
    # # remove those galaxies
    # for galaxy in bad_fit:
    #     all_properties = all_properties.drop(galaxy, axis=0)
    #
    #
    # # take only the elliptical galaxies
    # all_properties = all_properties[all_properties["n_r"] >= 4]
    #
    # # get a list of all the ids of the galaxies
    # chosen_galaxies = list(all_properties["GalaxyID"])
    #
    # # list to contain all galaxy images
    # all_images = []
    #
    # # # loop through each galaxy
    # for i, galaxy in enumerate(chosen_galaxies):
    #
    #     # get the filename of each galaxy in the supplemental file
    #     filename = "galrand_" + str(galaxy) + ".png"
    #
    #     # open the image and append it to the main list
    #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    #
    #     # normalise the image (each band independently)
    #     image = normalise_independently(image)
    #
    #     # add the image to the dataset
    #     all_images.append(image)
    #
    #
    # # split the data into training and testing data (12 images used for testing)
    # train_images = np.array(all_images[:-12])
    # test_images = np.array(all_images[-12:])
    #
    # print(train_images.shape)
    # print(test_images.shape)
    # print()







    # print(train_images.shape)
    # print(test_images.shape)











    # Define VAE model with custom train step
    class VAE(keras.Model):

        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return[
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        # custom train step
        def train_step(self, data):

            print("Data Shape:", data.shape)

            with tf.GradientTape() as tape:

                # get the latent representation (run image through the encoder)
                z_mean, z_log_var, z = self.encoder(data)

                print("Z Mean Shape:", z_mean.shape)
                print("Z Shape:", z.shape)

                # form the reconstruction (run latent representation through decoder)
                reconstruction = self.decoder(z)

                print("Reconstruction Shape:", reconstruction.shape)

                # calculate the binary cross entropy reconstruction loss (sum over each pixel and average (mean) across each channel and across the batch)
                # reconstruction_loss = ops.mean(
                #     ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2),
                #     )
                # )
                # reconstruction_loss = ops.mean(keras.losses.binary_crossentropy(data, reconstruction))
                # reconstruction_loss = ops.mean(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2))
                reconstruction_loss = ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2))
                reconstruction_loss = reconstruction_loss / (256 * 256)

                print("Reconstruction Loss Shape:", reconstruction_loss.shape)


                # calculate the kl divergence (sum over each latent feature and average (mean) across the batch)
                # kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                # kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
                # kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                # kl_loss = ops.mean(kl_loss, axis=1)
                kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                kl_loss = ops.sum(kl_loss, axis=1) / encoding_dim

                print("KL Loss Shape:", kl_loss.shape)


                # total loss is the sum of reconstruction loss and kl divergence
                total_loss = reconstruction_loss + kl_loss

                print("Total Loss Shape:", total_loss.shape)


            # gradient decent based on total loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # update loss trackers
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            # return total loss, reconstruction loss and kl divergence
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }






    # define sampling layer
    class Sampling(Layer):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, inputs):

            # get the latent distributions
            z_mean, z_log_var = inputs

            # find the batch size and number of latent features (dim)
            batch = ops.shape(z_mean)[0]
            dim = ops.shape(z_mean)[1]

            # generate the random variables
            epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)

            # perform reparameterization trick
            return z_mean + ops.exp(0.5 * z_log_var) * epsilon









    # Define keras tensor for the encoder
    input_image = keras.Input(shape=(256, 256, 3))                                                                  # (256, 256, 3)

    # layers for the encoder
    x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 32)
    x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 64)
    x = Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (32, 32, 128)
    x = Conv2D(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (16, 16, 256)
    x = Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (8, 8, 512)
    # x = Flatten()(x)                                                                                              # (8*8*512 = 32768)
    x = GlobalAveragePooling2D()(x)                                                                                 # (512)
    x = Dense(128, activation="relu")(x)                                                                            # (128)

    z_mean = Dense(encoding_dim, name="z_mean")(x)
    z_log_var = Dense(encoding_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    # build the encoder
    encoder = keras.Model(input_image, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()


    # Define keras tensor for the decoder
    latent_input = keras.Input(shape=(encoding_dim,))

    # layers for the decoder
    x = Dense(units=128, activation="relu")(latent_input)                                                           # (64)
    x = Dense(units=512, activation="relu")(x)                                                                      # (256)
    x = Dense(units=8*8*512, activation="relu")(x)                                                                  # (8*8*512 = 32768)
    x = Reshape((8, 8, 512))(x)                                                                                     # (8, 8, 512)
    x = Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (16, 16, 256)
    x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (32, 32, 128)
    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (64, 64, 64)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (128, 128, 32)
    decoded = Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="sigmoid", padding="same")(x)        # (256, 256, 3)

    # build the decoder
    decoder = keras.Model(latent_input, decoded, name="decoder")
    decoder.summary()




    # build and compile the VAE
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())




    # train the model
    model_loss = vae.fit(train_images, epochs=epochs, batch_size=batch_size)

    # or load the weights from a previous run
    # vae.load_weights("Variational Eagle/Weights/Normalised to r/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_1.weights.h5")


    # save the weights
    vae.save_weights(filepath="Variational Eagle/Weights/Fully Balanced Mean/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5", overwrite=True)

    # generate extracted features from trained encoder and save as numpy array
    extracted_features = vae.encoder.predict(train_images)
    np.save("Variational Eagle/Extracted Features/Fully Balanced Mean/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy", extracted_features)

    print(np.array(extracted_features).shape)

    # get loss, reconstruction loss and kl loss and save as numpy array
    loss = np.array([model_loss.history["loss"][-1], model_loss.history["reconstruction_loss"][-1], model_loss.history["kl_loss"][-1]])
    print("\n \n" + str(encoding_dim))
    print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")
    np.save("Variational Eagle/Loss/Fully Balanced Mean/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_loss_" + str(run) + ".npy", loss)




    # # loss plot for individual run
    # fig, axs1 = plt.subplots()
    # axs1.plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss")
    # axs1.set_ylabel("reconstruction loss")
    # axs2 = axs1.twinx()
    # axs2.plot(model_loss.history["kl_loss"], label="KL Loss", color="y")
    # axs2.set_ylabel("KL Loss")
    # plt.legend()
    #
    # plt.savefig("Variational Eagle/Loss Plots/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epochs_" + str(batch_size) + "_bs_loss_" + str(run))
    # plt.show()

    # log scale and normal


    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    axs[0].plot(model_loss.history["loss"], label="Total Loss", color="black")
    axs[0].plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    axs[0].plot(model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(np.log10(model_loss.history["loss"]), label="Total Loss", color="black")
    axs[1].plot(np.log10(model_loss.history["reconstruction_loss"]), label="Reconstruction Loss", color="C0")
    axs[1].plot(np.log10(model_loss.history["kl_loss"]), label="KL Divergence", color="C1")
    axs[1].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Log(Loss)")

    axs[2].plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    axs2 = axs[2].twinx()
    axs2.plot(model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    lines = axs[2].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    labels = axs[2].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    axs[2].legend(lines, labels)
    # axs[2].legend()
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Reconstruction Loss")
    axs2.set_ylabel("KL Divergence")

    plt.savefig("Variational Eagle/Loss Plots/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epochs_" + str(batch_size) + "_bs_loss_" + str(run))
    plt.show()









    # Training Reconstructions

    # number of images to reconstruct
    n = 12

    # create a subset of the validation data to reconstruct (first 10 images)
    images_to_reconstruct = train_images[n:]

    # reconstruct the images
    test_features, _, _ = vae.encoder.predict(images_to_reconstruct)
    reconstructed_images = vae.decoder.predict(test_features)

    # create figure to hold subplots
    fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))

    # plot each subplot
    for i in range(0, n - 1):
        original_image = normalise_independently(images_to_reconstruct[i])
        reconstructed_image = normalise_independently(reconstructed_images[i])

        # show the original image (remove axes)
        axs[0, i].imshow(original_image)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)

        # show the reconstructed image (remove axes)
        axs[1, i].imshow(reconstructed_image)
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)

    plt.savefig("Variational Eagle/Reconstructions/Training/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_reconstruction_" + str(run))
    plt.show()







    # Testing Reconstructions

    # number of images to reconstruct
    n = 12

    # create a subset of the validation data to reconstruct (first 10 images)
    images_to_reconstruct = test_images[:n]
    # images_to_reconstruct = train_images[n:]

    # reconstruct the images
    test_features, _, _ = vae.encoder.predict(images_to_reconstruct)
    reconstructed_images = vae.decoder.predict(test_features)

    # create figure to hold subplots
    fig, axs = plt.subplots(2, n-1, figsize=(18,5))

    # plot each subplot
    for i in range(0, n-1):

        original_image = normalise_independently(images_to_reconstruct[i])
        reconstructed_image = normalise_independently(reconstructed_images[i])

        # show the original image (remove axes)
        axs[0,i].imshow(original_image)
        axs[0,i].get_xaxis().set_visible(False)
        axs[0,i].get_yaxis().set_visible(False)

        # show the reconstructed image (remove axes)
        axs[1,i].imshow(reconstructed_image)
        axs[1,i].get_xaxis().set_visible(False)
        axs[1,i].get_yaxis().set_visible(False)

    plt.savefig("Variational Eagle/Reconstructions/Testing/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_reconstruction_" + str(run))
    plt.show()










# epochs = [25, 50, 100, 150, 200, 250, 300, 350, 400]
#
# for epoch in epochs:
#
#     # load the weights
#     vae.load_weights("Variational Eagle/Weights/" + str(encoding_dim) + "_feature_" + str(epoch) + "_epoch_weights.weights.h5")
#
#     # # save the weights
#     # vae.save_weights(filepath="Variational Eagle/Weights/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights.weights.h5", overwrite=True)
#     #
#     #
#     # # generate extracted features from trained encoder and save as numpy array
#     # extracted_features = vae.encoder.predict(train_images)
#     # np.save("Variational Eagle/Extracted Features/" + str(encoding_dim) + "feature_" + str(epochs) + "_epoch_features.npy", extracted_features)
#     #
#     #
#     # # get loss, reconstruction loss and kl loss and save as numpy array
#     # loss = np.array([model_loss.history["loss"][-1], model_loss.history["reconstruction_loss"][-1], model_loss.history["kl_loss"][-1]])
#     # print("\n \n" + str(encoding_dim))
#     # print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")
#     # np.save("Variational Eagle/Loss/" + str(encoding_dim) + "_feature" + str(epochs) + "_epoch_loss.npy", loss)
#
#
#
#
#     # # loss plot for individual run
#     # fig, axs1 = plt.subplots()
#     # axs1.plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss")
#     # axs1.set_ylabel("reconstruction loss")
#     # axs2 = axs1.twinx()
#     # axs2.plot(model_loss.history["kl_loss"], label="KL Loss", color="y")
#     # axs2.set_ylabel("KL Loss")
#     # plt.legend()
#     #
#     # plt.savefig("Variational Eagle/Plots/" + str(encoding_dim) + "_feature" + str(epochs) + "_epoch_loss")
#     # plt.show()
#
#
#
#
#     # number of images to reconstruct
#     n = 12
#
#     # create a subset of the validation data to reconstruct (first 10 images)
#     # images_to_reconstruct = test_images[n:]
#     images_to_reconstruct = train_images[n:]
#
#     # reconstruct the images
#     test_features, _, _ = vae.encoder.predict(images_to_reconstruct)
#     reconstructed_images = vae.decoder.predict(test_features)
#
#     # create figure to hold subplots
#     fig, axs = plt.subplots(2, n-1, figsize=(18,5))
#
#     # plot each subplot
#     for i in range(0, n-1):
#
#         # show the original image (remove axes)
#         axs[0,i].imshow(images_to_reconstruct[i])
#         axs[0,i].get_xaxis().set_visible(False)
#         axs[0,i].get_yaxis().set_visible(False)
#
#         # show the reconstructed image (remove axes)
#         axs[1,i].imshow(reconstructed_images[i])
#         axs[1,i].get_xaxis().set_visible(False)
#         axs[1,i].get_yaxis().set_visible(False)
#
#     plt.savefig("Variational Eagle/Reconstructions/Training/" + str(encoding_dim) + "_feature_" + str(epoch) + "_epoch_reconstruction_2")
#     plt.show()


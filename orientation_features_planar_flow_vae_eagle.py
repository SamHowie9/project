import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# from tensorflow.keras import models
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
# from tensorflow.keras import layers, models, backend as K, losses, optimizers, metrics, Input, ops
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics, losses, optimizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D, Layer, Input
from tensorflow.random import normal, Generator
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg



tf.keras.mixed_precision.set_global_policy('float32')


tfb = tfp.bijectors
tfd = tfp.distributions






encoding_dim = 30
beta = 0.0001
beta_name = "0001"
epochs = 750
n_flows = 3
run = 1
batch_size = 32


# select which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"






for n_flows in [6, 7, 8, 9, 10, 11, 12]:




    print()
    print("Encoding Dim", encoding_dim)
    print("Run", run)
    print("Beta", beta, beta_name)
    print("Epochs", epochs)
    print("Batch Size", batch_size)
    print("Flows", n_flows)
    print()



    # normalise each band individually
    def normalise_independently(image):
        image = image.T
        for i in range(0, 3):
            image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
        return image.T











    # load the images as a fully balanced dataset

    # load structural and physical properties into dataframes
    structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
    physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

    # dataframe for all properties
    all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

    # find all bad fit galaxies
    bad_fit = all_properties[((all_properties["flag_r"] == 1) |
                              (all_properties["flag_r"] == 4) |
                              (all_properties["flag_r"] == 5) |
                              (all_properties["flag_r"] == 6))].index.tolist()

    print("Bad Fit Indices:", bad_fit)
    print()

    # remove those galaxies
    for galaxy in bad_fit:
        all_properties = all_properties.drop(galaxy, axis=0)

    # get a list of all the ids of the galaxies
    chosen_galaxies = list(all_properties["GalaxyID"])


    # list to contain all galaxy images
    all_images_rand = []

    # # loop through each galaxy
    for i, galaxy in enumerate(chosen_galaxies):

        # get the filename of each galaxy in the supplemental file
        filename = "galrand_" + str(galaxy) + ".png"

        # open the image and append it to the main list
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

        # normalise the image (each band independently)
        image = normalise_independently(image)

        # add the image to the dataset
        all_images_rand.append(image)


    # split the data into training and testing data (200 images used for testing)
    train_images_rand = all_images_rand[:-200]
    test_images_rand = np.array(all_images_rand[-200:])





    # list to contain all galaxy images
    all_images_face = []

    # # loop through each galaxy
    for i, galaxy in enumerate(chosen_galaxies):

        # get the filename of each galaxy in the supplemental file
        filename = "galface_" + str(galaxy) + ".png"

        # open the image and append it to the main list
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

        # normalise the image (each band independently)
        image = normalise_independently(image)

        # add the image to the dataset
        all_images_face.append(image)


    # split the data into training and testing data (200 images used for testing)
    train_images_face = all_images_face[:-200]
    test_images_face = np.array(all_images_face[-200:])







    # list to contain all galaxy images
    all_images_edge = []

    # # loop through each galaxy
    for i, galaxy in enumerate(chosen_galaxies):
        # get the filename of each galaxy in the supplemental file
        filename = "galedge_" + str(galaxy) + ".png"

        # open the image and append it to the main list
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

        # normalise the image (each band independently)
        image = normalise_independently(image)

        # add the image to the dataset
        all_images_edge.append(image)

    # split the data into training and testing data (200 images used for testing)
    train_images_edge = all_images_edge[:-200]
    test_images_edge = np.array(all_images_edge[-200:])












    #load only the spiral galaxies

    # # load structural and physical properties into dataframes
    # structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
    # physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
    #
    # # dataframe for all properties
    # all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
    #
    # # find all bad fit galaxies
    # bad_fit = all_properties[((all_properties["flag_r"] == 1) |
    #                           (all_properties["flag_r"] == 4) |
    #                           (all_properties["flag_r"] == 5) |
    #                           (all_properties["flag_r"] == 6))].index.tolist()
    #
    # print("Bad Fit Indices:", bad_fit)
    # print()
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
    class VAE(Model):

        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = metrics.Mean(name="kl_loss")

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
                z_mean, z_log_var, z, sum_log_det_jacobians  = self.encoder(data)

                # form the reconstruction (run latent representation through decoder)
                reconstruction = self.decoder(z)

                # reconstruction loss
                reconstruction_loss = tf.reduce_mean(losses.binary_crossentropy(data, reconstruction))

                # kl loss
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_sum(kl_loss, axis=1) - sum_log_det_jacobians
                kl_loss = tf.reduce_mean(kl_loss)

                # total loss
                # total_loss = reconstruction_loss + kl_loss
                total_loss = reconstruction_loss + (beta * kl_loss)



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

        def __init__(self, latent_dim, n_flows=1, **kwargs):
            super().__init__(**kwargs)
            self.latent_dim = latent_dim
            self.n_flows = n_flows
            self.flows = [PlanarFlow(latent_dim) for _ in range(n_flows)]

        def call(self, inputs):

            # get the latent distributions
            z_mean, z_log_var = inputs

            # find the batch size and number of latent features (dim)
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]

            # generate the random variables
            epsilon = tf.random.normal(shape=(batch, dim))

            # perform reparameterization trick
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

            z = tf.clip_by_value(z, -4+1e-4, 4-1e-4)

            sum_log_det_jacobian = 0.0

            for flow in self.flows:
                z, log_det = flow(z)
                sum_log_det_jacobian += log_det


            return z, sum_log_det_jacobian






    class PlanarFlow(Layer):

        def __init__(self, latent_dim, **kwargs):
            super().__init__(**kwargs)
            self.latent_dim = latent_dim

        def build(self, input_shape):

            # Initialize weights and bias for the planar transformation
            self.u = self.add_weight(shape=(self.latent_dim,), initializer='random_normal', trainable=True)
            self.w = self.add_weight(shape=(self.latent_dim,), initializer='random_normal', trainable=True)
            self.b = self.add_weight(shape=(), initializer='zeros', trainable=True)

        def call(self, z):

            # parameterization of u (ensure eTu > -1)
            u_hat = self.u + (tf.nn.softplus(tf.reduce_sum(self.w * self.u)) - 1 - tf.reduce_sum(self.w * self.u)) * self.w / (tf.norm(self.w) ** 2 + 1e-8)

            # transformation
            w_dot_z = tf.reduce_sum(self.w * z, axis=1, keepdims=True)
            activation = tf.tanh(w_dot_z + self.b)
            z_transformed = z + (u_hat * activation)

            # calculate derivative
            psi = (1.0 - tf.square(activation)) * self.w

            # find the log det jacobian
            det_jacobian = 1.0 + tf.reduce_sum(psi * u_hat, axis=1)  # shape: (batch_size,)
            log_det_jacobian = tf.math.log(tf.abs(det_jacobian) + 1e-8)  # add epsilon for numerical stability

            return z_transformed, log_det_jacobian






    # apply the flows to the latent vectors after training
    def apply_flows(z_mean, flows):

        # convert vectors to tensor and clip (as done in sampling layer)
        z = tf.convert_to_tensor(z_mean, dtype=tf.float32)
        z = tf.clip_by_value(z, -4+1e-4, 4-1e-4)

        # apply the flows
        for flow in flows:
            z, _ = flow(z)

        # return the transformed vector
        return z





    # Define keras tensor for the encoder
    input_image = Input(shape=(256, 256, 3))                                                                  # (256, 256, 3)

    # layers for the encoder
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 32)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 64)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (32, 32, 128)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (16, 16, 256)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (8, 8, 512)
    # x = Flatten()(x)                                                                                              # (8*8*512 = 32768)
    x = layers.GlobalAveragePooling2D()(x)                                                                                 # (512)
    x = layers.Dense(128, activation="relu")(x)                                                                            # (128)

    z_mean = layers.Dense(encoding_dim, name="z_mean")(x)
    z_log_var = layers.Dense(encoding_dim, name="z_log_var")(x)

    z, sum_log_det_jacobians = Sampling(encoding_dim, n_flows=n_flows)([z_mean, z_log_var])

    # build the encoder
    encoder = Model(input_image, [z_mean, z_log_var, z, sum_log_det_jacobians], name="encoder")
    encoder.summary()



    # Define keras tensor for the decoder
    latent_input = Input(shape=(encoding_dim,))

    # layers for the decoder
    x = layers.Dense(units=128, activation="relu")(latent_input)                                                           # (64)
    x = layers.Dense(units=512, activation="relu")(x)                                                                      # (256)
    x = layers.Dense(units=8*8*512, activation="relu")(x)                                                                  # (8*8*512 = 32768)
    x = layers.Reshape((8, 8, 512))(x)                                                                                     # (8, 8, 512)
    x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (16, 16, 256)
    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (32, 32, 128)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (64, 64, 64)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (128, 128, 32)
    decoded = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="sigmoid", padding="same")(x)        # (256, 256, 3)

    # build the decoder
    decoder = Model(latent_input, decoded, name="decoder")
    decoder.summary()



    # build and compile the VAE
    vae = VAE(encoder, decoder)
    # vae.compile(optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0))
    # vae.compile(optimizer=optimizers.Adam(clipnorm=1.0))
    vae.compile(optimizer=optimizers.Adam())



    # train the model
    # model_loss = vae.fit(train_images, epochs=epochs, batch_size=batch_size)

    # get loss, reconstruction loss and kl loss and save as numpy array
    # loss = np.array([model_loss.history["loss"][-1], model_loss.history["reconstruction_loss"][-1], model_loss.history["kl_loss"][-1]])
    # print("\n \n" + str(encoding_dim))
    # print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")
    # np.save("Variational Eagle/Loss/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", loss)




    vae.build(input_shape=(None, 256, 256, 3))

    # or load the weights from a previous run
    vae.load_weights("Variational Eagle/Weights/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")


    # save the weights
    # vae.save_weights(filepath="Variational Eagle/Weights/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5", overwrite=True)





    # number of images to reconstruct
    n = 12





    # face on orientation

    # generate extracted features from trained encoder and save as numpy array
    z_mean, _, _, _ = vae.encoder.predict(train_images_face)

    np.save("Variational Eagle/Extracted Features/Normalising Flow/Face On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", z_mean)

    # get the sampling layer
    sampling_layer = None
    for layer in encoder.layers:
        if isinstance(layer, Sampling):
            sampling_layer = layer
            break

    # get the flows from the sampling layer
    flows = sampling_layer.flows

    # transform the mean vectors
    z_transformed = apply_flows(z_mean, flows)

    np.save("Variational Eagle/Extracted Features/Normalising Flow/Face On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy", z_transformed)


    # create a subset of the validation data to reconstruct (first 10 images)
    images_to_reconstruct = train_images_face[n:]


    # reconstruct the original vectors
    reconstructed_images = vae.decoder.predict(z_mean[n:])

    # create figure to hold subplots
    fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))

    # plot each subplot
    for i in range(0, n - 1):
        # normalise the images
        original_image = normalise_independently(images_to_reconstruct[i])
        reconstructed_image = normalise_independently([i])

        # show the original image (remove axes)
        axs[0, i].imshow(original_image)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)

        # show the reconstructed image (remove axes)
        axs[1, i].imshow(reconstructed_image)
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)

    plt.savefig("Variational Eagle/Reconstructions/Training/Normalising Flow/Face On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    plt.show()


    # reconstruct the transformed vectors images
    reconstructed_images = vae.decoder.predict(z_transformed[n:])

    # create figure to hold subplots
    fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))

    # plot each subplot
    for i in range(0, n - 1):
        # normalise the images
        original_image = normalise_independently(images_to_reconstruct[i])
        reconstructed_image = normalise_independently([i])

        # show the original image (remove axes)
        axs[0, i].imshow(original_image)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)

        # show the reconstructed image (remove axes)
        axs[1, i].imshow(reconstructed_image)
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)

    plt.savefig("Variational Eagle/Reconstructions/Training/Normalising Flow/Face On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed")
    plt.show()












    # edge on

    # generate extracted features from trained encoder and save as numpy array
    z_mean, _, _, _ = vae.encoder.predict(train_images_face)

    np.save("Variational Eagle/Extracted Features/Normalising Flow/Edge On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", z_mean)

    # get the sampling layer
    sampling_layer = None
    for layer in encoder.layers:
        if isinstance(layer, Sampling):
            sampling_layer = layer
            break

    # get the flows from the sampling layer
    flows = sampling_layer.flows

    # transform the mean vectors
    z_transformed = apply_flows(z_mean, flows)

    np.save("Variational Eagle/Extracted Features/Normalising Flow/Edge On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy", z_transformed)

    # create a subset of the validation data to reconstruct (first 10 images)
    images_to_reconstruct = train_images_face[n:]

    # reconstruct the original vectors
    reconstructed_images = vae.decoder.predict(z_mean[n:])

    # create figure to hold subplots
    fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))

    # plot each subplot
    for i in range(0, n - 1):
        # normalise the images
        original_image = normalise_independently(images_to_reconstruct[i])
        reconstructed_image = normalise_independently([i])

        # show the original image (remove axes)
        axs[0, i].imshow(original_image)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)

        # show the reconstructed image (remove axes)
        axs[1, i].imshow(reconstructed_image)
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)

    plt.savefig("Variational Eagle/Reconstructions/Training/Normalising Flow/Edge On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    plt.show()

    # reconstruct the transformed vectors images
    reconstructed_images = vae.decoder.predict(z_transformed[n:])

    # create figure to hold subplots
    fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))

    # plot each subplot
    for i in range(0, n - 1):
        # normalise the images
        original_image = normalise_independently(images_to_reconstruct[i])
        reconstructed_image = normalise_independently([i])

        # show the original image (remove axes)
        axs[0, i].imshow(original_image)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)

        # show the reconstructed image (remove axes)
        axs[1, i].imshow(reconstructed_image)
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)

    plt.savefig("Variational Eagle/Reconstructions/Training/Normalising Flow/Edge On/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed")
    plt.show()
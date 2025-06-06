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
run = 1
n_flows = 2
beta = 0.0001
beta_name = "0001"

# select which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="9"


# number of epochs for run
epochs = 500

# batch size for run
batch_size = 32






for encoding_dim in [encoding_dim]:



    print()
    print("Encoding Dim", encoding_dim)
    print("Run", run)
    print("Beta", beta, beta_name)
    print("Epochs", epochs)
    print("Batch Size", batch_size)
    print()



    # normalise each band individually
    def normalise_independently(image):
        image = image.T
        for i in range(0, 3):
            image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
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

    print("Original Dataset", len(all_images))

    # split the data into training and testing data (200 images used for testing)
    train_images = all_images[:-200]
    test_images = np.array(all_images[-200:])

    print("Training Set", len(train_images))
    print("Testing Set", len(test_images))
    print()


    # load the filenames of the augmented elliptical images
    augmented_galaxies =  os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/")

    print("Augmented Ellipticals", len(augmented_galaxies))

    for galaxy in augmented_galaxies:

        # load each augmented image
        image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/" + galaxy)

        # normalise the image
        image = normalise_independently(image)

        # add the image to the training set (not the testing set)
        train_images.append(image)

    print("Training Set", len(train_images))
    print()


    # load the filenames of the augmented unknown images
    augmented_galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/")

    print("Augmented Unknown", len(augmented_galaxies))


    for galaxy in augmented_galaxies:

        # load each augmented image
        image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/" + galaxy)

        # normalise the image
        image = normalise_independently(image)

        # add the image to the training set (not the testing set)
        train_images.append(image)

    # convert the training set to a numpy array
    train_images = np.array(train_images)


    print("Training Set", train_images.shape)
    print("Testing Set", test_images.shape)
    print()







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
            self.flows = [RQSFlow(latent_dim) for _ in range(n_flows)]

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






    class RQSFlow(Layer):

        def __init__(self, latent_dim, num_bins=4, bound=4.0, **kwargs):
            super().__init__(**kwargs)
            self.latent_dim = latent_dim
            self.num_bins = num_bins
            self.bound = bound
            self.bijectors = []

        def build(self, input_shape):
            # Simple neural network to predict spline parameters per dimension
            self.nn = tf.keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dense(self.latent_dim * (3 * self.num_bins - 1))
            ])

        def call(self, z):

            # clip z to make sure it is within the allowed bounds
            # z = tf.clip_by_value(z, -self.bound + 1e-4, self.bound - 1e-4)

            # get the batch size
            batch_size = tf.shape(z)[0]

            # Predict parameters for the splines
            params = self.nn(z)
            params = tf.reshape(params, [batch_size, self.latent_dim, 3 * self.num_bins - 1])

            # Unpack RQS parameters: widths, heights, slopes
            unnormalized_widths = params[:, :, :self.num_bins]
            unnormalized_heights = params[:, :, self.num_bins:2 * self.num_bins]
            unnormalized_derivatives = params[:, :, 2 * self.num_bins:]

            # Apply constraints for monotonicity
            widths = tf.nn.softmax(unnormalized_widths, axis=-1)
            heights = tf.nn.softmax(unnormalized_heights, axis=-1)
            derivatives = tf.nn.softplus(unnormalized_derivatives) + 1e-3  # to ensure positivity
            derivatives = tf.clip_by_value(derivatives, 1e-2, 1e2)

            # Create bijectors per dimension
            z_out = []
            log_dets = []

            for d in range(self.latent_dim):
                rqs_bijector = tfb.RationalQuadraticSpline(
                    bin_widths=widths[:, d, :],
                    bin_heights=heights[:, d, :],
                    knot_slopes=derivatives[:, d, :],
                    range_min=-self.bound,
                )

                z_d = z[:, d]
                z_d_transformed = rqs_bijector.forward(z_d)
                # z_d_transformed = tf.clip_by_value(z_d_transformed, -self.bound + 1e-4, self.bound - 1e-4)
                log_det = rqs_bijector.forward_log_det_jacobian(z_d, event_ndims=0)

                tf.debugging.check_numerics(log_det, message="NaN in log_det")
                tf.debugging.check_numerics(z_d_transformed, message="NaN in z_d_transformed")

                z_out.append(z_d_transformed)
                log_dets.append(log_det)

            # Stack outputs and sum log-determinants
            z_out = tf.stack(z_out, axis=-1)
            sum_log_det = tf.add_n(log_dets)

            return z_out, sum_log_det






    # class PlanarFlow(Layer):
    #     def __init__(self, latent_dim, **kwargs):
    #         super().__init__(**kwargs)
    #         self.latent_dim = latent_dim
    #
    #     def build(self, input_shape):
    #         self.u = self.add_weight(shape=(self.latent_dim, 1), initializer='random_normal', trainable=True, name='u')
    #         self.w = self.add_weight(shape=(self.latent_dim, 1), initializer='random_normal', trainable=True, name='w')
    #         self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='b')
    #
    #     def call(self, z0):
    #
    #         wTz = tf.matmul(z0, self.w) + self.b  # shape (batch, 1)
    #         activation = tf.tanh(wTz)  # shape (batch, 1)
    #         z = z0 + tf.matmul(activation, tf.transpose(self.u))  # (batch, latent_dim)
    #
    #         # Compute psi: shape (batch, latent_dim)
    #         tanh_derivative = 1 - tf.square(tf.tanh(wTz))  # shape (batch, 1)
    #         psi = tanh_derivative * tf.transpose(self.w)  # shape (batch, latent_dim)
    #
    #         # Compute dot product for log-det Jacobian: uᵀ ψ, shape (batch,)
    #         dot = tf.reduce_sum(psi * tf.transpose(self.u), axis=1)  # shape (batch,)
    #         log_det_jacobian = tf.math.log(tf.abs(1 + dot) + 1e-7)  # shape (batch,)
    #         return z, log_det_jacobian







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
    vae.compile(optimizer=optimizers.Adam())
    # vae.compile(optimizer=optimizers.Adam(clipvalue=1.0))
    # vae.compile(optimizer=optimizers.Adam())



    # train the model
    model_loss = vae.fit(train_images, epochs=epochs, batch_size=batch_size)

    # or load the weights from a previous run
    # vae.load_weights("Variational Eagle/Weights/Normalised to r/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_1.weights.h5")

    vae.build(input_shape=(None, 256, 256, 3))

    # save the weights
    vae.save_weights(filepath="Variational Eagle/Weights/Normalising Flow/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5", overwrite=True)


    # generate extracted features from trained encoder and save as numpy array
    extracted_features = vae.encoder.predict(train_images)
    np.save("Variational Eagle/Extracted Features/Normalising Flow/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", extracted_features[0:3])

    # print(np.array(extracted_features).shape)

    # get loss, reconstruction loss and kl loss and save as numpy array
    loss = np.array([model_loss.history["loss"][-1], model_loss.history["reconstruction_loss"][-1], model_loss.history["kl_loss"][-1]])
    print("\n \n" + str(encoding_dim))
    print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")
    np.save("Variational Eagle/Loss/Normalising Flow/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", loss)






    # loss plot for run

    fig, axs = plt.subplots(3, 3, figsize=(30, 15))

    axs[0][0].plot(model_loss.history["loss"], label="Total Loss", color="black")
    axs[0][0].plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    axs[0][0].plot(model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    axs[0][0].legend()
    axs[0][0].set_xlabel("Epoch")
    axs[0][0].set_ylabel("Loss")

    axs[0][1].plot(model_loss.history["loss"][:10], label="Total Loss", color="black")
    axs[0][1].plot(model_loss.history["reconstruction_loss"][:10], label="Reconstruction Loss", color="C0")
    axs[0][1].plot(model_loss.history["kl_loss"][:10], label="KL Divergence", color="C1")
    axs[0][1].legend()
    axs[0][1].set_xlabel("Epoch")
    axs[0][1].set_ylabel("Loss")

    axs[0][2].plot(model_loss.history["loss"][50:], label="Total Loss", color="black")
    axs[0][2].plot(model_loss.history["reconstruction_loss"][50:], label="Reconstruction Loss", color="C0")
    axs[0][2].plot(model_loss.history["kl_loss"][50:], label="KL Divergence", color="C1")
    axs[0][2].legend()
    axs[0][2].set_xlabel("Epoch")
    axs[0][2].set_ylabel("Loss")



    axs[1][0].plot(np.log10(model_loss.history["loss"]), label="Total Loss", color="black")
    axs[1][0].plot(np.log10(model_loss.history["reconstruction_loss"]), label="Reconstruction Loss", color="C0")
    axs[1][0].plot(np.log10(model_loss.history["kl_loss"]), label="KL Divergence", color="C1")
    axs[1][0].legend()
    axs[1][0].set_xlabel("Epoch")
    axs[1][0].set_ylabel("Log(Loss)")

    axs[1][1].plot(np.log10(model_loss.history["loss"][:10]), label="Total Loss", color="black")
    axs[1][1].plot(np.log10(model_loss.history["reconstruction_loss"][:10]), label="Reconstruction Loss", color="C0")
    axs[1][1].plot(np.log10(model_loss.history["kl_loss"][:10]), label="KL Divergence", color="C1")
    axs[1][1].legend()
    axs[1][1].set_xlabel("Epoch")
    axs[1][1].set_ylabel("Log(Loss)")

    axs[1][2].plot(np.log10(model_loss.history["loss"][50:]), label="Total Loss", color="black")
    axs[1][2].plot(np.log10(model_loss.history["reconstruction_loss"][50:]), label="Reconstruction Loss", color="C0")
    axs[1][2].plot(np.log10(model_loss.history["kl_loss"][50:]), label="KL Divergence", color="C1")
    axs[1][2].legend()
    axs[1][2].set_xlabel("Epoch")
    axs[1][2].set_ylabel("Log(Loss)")



    axs[2][0].plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    axs2 = axs[2][0].twinx()
    axs2.plot(model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    lines = axs[2][0].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    labels = axs[2][0].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    axs[2][0].legend(lines, labels)
    # axs[2].legend()
    axs[2][0].set_xlabel("Epoch")
    axs[2][0].set_ylabel("Reconstruction Loss")
    axs2.set_ylabel("KL Divergence")

    axs[2][1].plot(model_loss.history["reconstruction_loss"][:10], label="Reconstruction Loss", color="C0")
    axs2 = axs[2][1].twinx()
    axs2.plot(model_loss.history["kl_loss"][:10], label="KL Divergence", color="C1")
    lines = axs[2][1].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    labels = axs[2][1].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    axs[2][1].legend(lines, labels)
    # axs[2].legend()
    axs[2][1].set_xlabel("Epoch")
    axs[2][1].set_ylabel("Reconstruction Loss")
    axs2.set_ylabel("KL Divergence")

    axs[2][2].plot(model_loss.history["reconstruction_loss"][50:], label="Reconstruction Loss", color="C0")
    axs2 = axs[2][2].twinx()
    axs2.plot(model_loss.history["kl_loss"][50:], label="KL Divergence", color="C1")
    lines = axs[2][2].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    labels = axs[2][2].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    axs[2][2].legend(lines, labels)
    # axs[2].legend()
    axs[2][2].set_xlabel("Epoch")
    axs[2][2].set_ylabel("Reconstruction Loss")
    axs2.set_ylabel("KL Divergence")



    # plt.savefig("Variational Eagle/Loss Plots/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epochs_" + str(batch_size) + "_bs_loss_" + str(run))
    plt.savefig("Variational Eagle/Loss Plots/Normalising Flows/normalising_flow_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    plt.show()






    # fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    #
    # axs[0].plot(model_loss.history["loss"], label="Total Loss", color="black")
    # axs[0].plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    # axs[0].plot(model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    # axs[0].legend()
    # axs[0].set_xlabel("Epoch")
    # axs[0].set_ylabel("Loss")
    #
    # axs[1].plot(np.log10(model_loss.history["loss"]), label="Total Loss", color="black")
    # axs[1].plot(np.log10(model_loss.history["reconstruction_loss"]), label="Reconstruction Loss", color="C0")
    # axs[1].plot(np.log10(model_loss.history["kl_loss"]), label="KL Divergence", color="C1")
    # axs[1].legend()
    # axs[1].set_xlabel("Epoch")
    # axs[1].set_ylabel("Log(Loss)")
    #
    # axs[2].plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    # axs2 = axs[2].twinx()
    # axs2.plot(model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    # lines = axs[2].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    # labels = axs[2].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    # axs[2].legend(lines, labels)
    # # axs[2].legend()
    # axs[2].set_xlabel("Epoch")
    # axs[2].set_ylabel("Reconstruction Loss")
    # axs2.set_ylabel("KL Divergence")
    #
    # # plt.savefig("Variational Eagle/Loss Plots/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epochs_" + str(batch_size) + "_bs_loss_" + str(run))
    # plt.savefig("Variational Eagle/Loss Plots/test_min_normal")
    # plt.show()


    # zoom in on loss plot and place to the right (epoch 1-10)








    # Training Reconstructions

    # number of images to reconstruct
    n = 12

    # create a subset of the validation data to reconstruct (first 10 images)
    images_to_reconstruct = train_images[n:]

    # reconstruct the images
    test_features, _, _, _ = vae.encoder.predict(images_to_reconstruct)
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

    # plt.savefig("Variational Eagle/Reconstructions/Training/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_reconstruction_" + str(run))
    plt.savefig("Variational Eagle/Reconstructions/Training/Normalising Flow/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    plt.show()







    # Testing Reconstructions

    # number of images to reconstruct
    n = 12

    # create a subset of the validation data to reconstruct (first 10 images)
    images_to_reconstruct = test_images[:n]
    # images_to_reconstruct = train_images[n:]

    # reconstruct the images
    test_features, _, _, _ = vae.encoder.predict(images_to_reconstruct)
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

    # plt.savefig("Variational Eagle/Reconstructions/Testing/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_reconstruction_" + str(run))
    plt.savefig("Variational Eagle/Reconstructions/Testing/Normalising Flow/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
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


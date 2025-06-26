import os
os.environ["KERAS_BACKEND"] = "tensorflow"
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






run = 2
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32


# select which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"






for encoding_dim in [encoding_dim]:
# for n_flows in [1, 2, 3]:
# for encoding_dim, n_flows in [[encoding_dim, 1], [encoding_dim, 2], [encoding_dim, 3], [encoding_dim+1, 1], [encoding_dim+1, 2], [encoding_dim+1, 3]]:
# for encoding_dim in [encoding_dim, encoding_dim+1, encoding_dim+2, encoding_dim+3, encoding_dim+4, encoding_dim+5]:
# for encoding_dim in range(encoding_dim, encoding_dim+6):
# for encoding_dim in range(encoding_dim, encoding_dim+5):
# for encoding_dim in [49, 50]:
# for encoding_dim in [encoding_dim]:
# for run in [1, 2, 3]:
# for run in range(run, run+14):



    print("\n \n")
    print("Encoding Dim", encoding_dim)
    print("Run", run)
    print("Flows", n_flows)
    print("Beta", beta, beta_name)
    print("Epochs", epochs)
    print("Batch Size", batch_size)
    print("\n \n")



    # normalise each band individually
    def normalise_independently(image):
        image = image.T
        for i in range(0, 3):
            image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
        return image.T








    # load the images as a balanced dataset (D/T)

    # chosen_galaxies = np.load("Galaxy Properties/Eagle Properties/chosen_glaxies.npy")
    #
    # # list to contain all galaxy images
    # all_images = []
    #
    # # loop through each galaxy
    # for i, galaxy in enumerate(chosen_galaxies):
    #
    #     # open the image and append it to the main list
    #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    #
    #     # normalise the image (each band independently)
    #     image = normalise_independently(image)
    #
    #     # add the image to the dataset
    #     all_images.append(image)
    #
    # print("Original Dataset", len(all_images))
    #
    # # split the data into training and testing data (200 images used for testing)
    # train_images = all_images
    # # train_images = all_images[:-200]
    # # test_images = np.array(all_images[-200:])
    #
    # # print("Training Set", len(train_images))
    # # print("Testing Set", len(test_images))
    # # print()
    #
    #
    #
    # # load the filenames of the augmented elliptical images
    # augmented_galaxies =  os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")
    #
    # print("Augmented Ellipticals", len(augmented_galaxies))
    #
    # for galaxy in augmented_galaxies:
    #
    #     # load each augmented image
    #     image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/" + galaxy)
    #
    #     # normalise the image
    #     image = normalise_independently(image)
    #
    #     # add the image to the training set (not the testing set)
    #     train_images.append(image)
    #
    #
    #
    # # load the filenames of the augmented transitional images
    # augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/")
    #
    # print("Augmented Transitional", len(augmented_galaxies))
    #
    # for galaxy in augmented_galaxies:
    #
    #     # load each augmented image
    #     image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/" + galaxy)
    #
    #     # normalise the image
    #     image = normalise_independently(image)
    #
    #     # add the image to the training set (not the testing set)
    #     train_images.append(image)
    #
    # # convert the training set to a numpy array
    # train_images = np.array(train_images)
    #
    #
    # print("Training Set", train_images.shape)
    # # print("Testing Set", test_images.shape)
    # print()












    # load spirals only

    spirals = np.load("Galaxy Properties/Eagle Properties/chosen_glaxies_spirals.npy")

    all_images = []

    for galaxy in spirals:

        # open the image and append it to the main list
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")

        # normalise the image (each band independently)
        image = normalise_independently(image)

        # add the image to the dataset
        all_images.append(image)


    augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Spirals Only/")

    for galaxy in augmented_galaxies:

        # load each augmented image
        image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Spirals Only/" + galaxy)

        # normalise the image
        image = normalise_independently(image)

        # add the image to the training set (not the testing set)
        all_images.append(image)


    train_images = np.array(all_images)
    # test_images = np.array(all_images[-200:])

    print("Spirals Training Set:", train_images.shape)







    # load transitional only

    # transitional = np.load("Galaxy Properties/Eagle Properties/chosen_glaxies_transitional.npy")
    #
    # all_images = []
    #
    # for galaxy in transitional:
    #     # open the image and append it to the main list
    #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    #
    #     # normalise the image (each band independently)
    #     image = normalise_independently(image)
    #
    #     # add the image to the dataset
    #     all_images.append(image)
    #
    # augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional Only/")
    #
    # for galaxy in augmented_galaxies:
    #
    #     # load each augmented image
    #     image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional Only/" + galaxy)
    #
    #     # normalise the image
    #     image = normalise_independently(image)
    #
    #     # add the image to the training set (not the testing set)
    #     all_images.append(image)
    #
    # train_images = np.array(all_images)
    # # test_images = np.array(all_images[-200:])
    #
    # print("Transitional Training Set:", train_images.shape)









    # load elliptical only

    # ellipticals = np.load("Galaxy Properties/Eagle Properties/chosen_glaxies_ellipticals.npy")
    #
    # all_images = []
    #
    # for galaxy in ellipticals:
    #     # open the image and append it to the main list
    #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    #
    #     # normalise the image (each band independently)
    #     image = normalise_independently(image)
    #
    #     # add the image to the dataset
    #     all_images.append(image)
    #
    # augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals Only/")
    #
    # for galaxy in augmented_galaxies:
    #
    #     # load each augmented image
    #     image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals Only/" + galaxy)
    #
    #     # normalise the image
    #     image = normalise_independently(image)
    #
    #     # add the image to the training set (not the testing set)
    #     all_images.append(image)
    #
    # train_images = np.array(all_images)
    # # test_images = np.array(all_images[-200:])
    #
    # print("Ellipticals Training Set:", train_images.shape)







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

            with tf.GradientTape() as tape:

                # get the latent representation (run image through the encoder)
                z_mean, z_log_var, z, sum_log_det_jacobians  = self.encoder(data)

                print("Z Shape", z_mean.shape, z.shape)

                # form the reconstruction (run latent representation through decoder)
                reconstruction = self.decoder(z)

                # reconstruction loss
                reconstruction_loss = tf.reduce_mean(losses.binary_crossentropy(data, reconstruction))

                # kl loss
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = (tf.reduce_sum(kl_loss, axis=1) - sum_log_det_jacobians) / z.shape[1]
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

            # perform reparameterisation trick
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

            # initialise as a tensor of batch size shape (same shape as first latent feature)
            sum_log_det_jacobian = tf.zeros_like(z_mean[:, 0])

            # apply flow transformations
            for flow in self.flows:
                z, log_det = flow(z)
                sum_log_det_jacobian += log_det

            return z, sum_log_det_jacobian





    # define planar flows
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

            # derivative of flow function
            psi = (1.0 - tf.square(activation)) * self.w

            # compute the log det jacobian
            det_jacobian = 1.0 + tf.reduce_sum(psi * u_hat, axis=1)  # shape: (batch_size,)
            log_det_jacobian = tf.math.log(tf.abs(det_jacobian) + 1e-8)  # add epsilon for numerical stability

            return z_transformed, log_det_jacobian





    # apply the flows to the latent vectors after training
    def apply_flows(z_mean, flows):

        # convert vectors to tensor and clip (as done in sampling layer)
        z = tf.convert_to_tensor(z_mean, dtype=tf.float32)
        z = tf.clip_by_value(z, -4 + 1e-4, 4 - 1e-4)

        sum_log_det_jacobian = 0.0

        # apply the flows
        for flow in flows:
            z, log_det = flow(z)
            sum_log_det_jacobian += log_det

        return z, sum_log_det_jacobian





    # Define keras tensor for the encoder
    input_image = Input(shape=(256, 256, 3))                                                                               # (256, 256, 3)

    # layers for the encoder
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 32)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 64)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (32, 32, 128)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (16, 16, 256)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (8, 8, 512)
    # x = Flatten()(x)                                                                                                     # (8*8*512 = 32768)
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
    decoded = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="sigmoid", padding="same")(x)         # (256, 256, 3)

    # build the decoder
    decoder = Model(latent_input, decoded, name="decoder")
    decoder.summary()



    # build and compile the VAE
    vae = VAE(encoder, decoder)
    # vae.compile(optimizer=optimizers.Adam(clipnorm=1.0))
    vae.compile(optimizer=optimizers.Adam())



    # train the model
    model_loss = vae.fit(train_images, epochs=epochs, batch_size=batch_size)

    # # get loss, reconstruction loss and kl loss and save as numpy array
    # loss = np.array([model_loss.history["loss"][-1], model_loss.history["reconstruction_loss"][-1], model_loss.history["kl_loss"][-1]])
    # np.save("Variational Eagle/Loss/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", loss)
    #
    # print("\n \n" + str(encoding_dim) + "   " + str(n_flows) + "   " + str(run))
    # print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")



    vae.build(input_shape=(None, 256, 256, 3))

    # or load the weights from a previous run
    # vae.load_weights("Variational Eagle/Weights/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")

    # save the weights
    vae.save_weights(filepath="Variational Eagle/Weights/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5", overwrite=True)



    # get and save the extracted features (pre transformations)
    z_mean, z_log_var, _, _ = vae.encoder.predict(train_images)
    np.save("Variational Eagle/Extracted Features/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", z_mean)



    # if we have flows
    if n_flows > 0:

        # get the sampling layer
        sampling_layer = None
        for layer in encoder.layers:
            if isinstance(layer, Sampling):
                sampling_layer = layer
                break

        # get the flows from the sampling layer
        flows = sampling_layer.flows

        # transform and save the mean vectors
        z_transformed, sum_log_det_jacobians = apply_flows(z_mean, flows)
        z_transformed = z_transformed.numpy()
        sum_log_det_jacobians = sum_log_det_jacobians.numpy()

    # if there are no flows
    else:
        z_transformed = z_mean
        sum_log_det_jacobians = np.zeros(z_mean.shape[0], dtype=np.float32)


    np.save("Variational Eagle/Extracted Features/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy", z_transformed)



    # reconstruct the image
    reconstructed_images = vae.decoder.predict(z_transformed)

    # use the CPU for this rather than GPU
    with tf.device('/CPU:0'):

        # get the reconstruction loss
        reconstruction_loss = tf.reduce_mean(losses.binary_crossentropy(train_images, reconstructed_images)).numpy().item()

        # # kl loss
        # kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        # kl_loss = (tf.reduce_sum(kl_loss, axis=1) - sum_log_det_jacobians) / z_mean.shape[1]
        # kl_loss = tf.reduce_mean(kl_loss)

        # kl loss
        kl_loss = -0.5 * (1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
        kl_loss = (np.sum(kl_loss, axis=1) - sum_log_det_jacobians) / z_mean.shape[1]
        kl_loss = kl_loss.mean()

        # total loss
        total_loss = reconstruction_loss + (beta * kl_loss)

        loss = np.array([total_loss, reconstruction_loss, kl_loss])
        np.save("Variational Eagle/Loss/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy", loss)

        print("\n \n" + str(encoding_dim) + "   " + str(n_flows) + "   " + str(run))
        print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")






    # number of images to reconstruct
    n = 12


    # n=12
    #
    # fig, axs = plt.subplots(12, n, figsize=(20, 20))
    #
    # # random.seed(0)
    # # reconstruction_indices = random.sample(range(train_images.shape[0]), n)
    # # images_to_reconstruct = train_images[reconstruction_indices]
    #
    # for i in range(0, 12):
    #
    #     random.seed(i)
    #     reconstruction_indices = random.sample(range(train_images.shape[0]), n)
    #     images_to_reconstruct = train_images[reconstruction_indices]
    #
    #     for j in range(0, n):
    #
    #         axs[i][j].imshow(images_to_reconstruct[j])
    #         axs[i][j].set_title(i)
    #
    # plt.savefig("Variational Eagle/Plots/reconstruction_random_sample")
    # plt.show()



    # training reconstructions

    # create a subset of the training data to reconstruct (first n images)
    # images_to_reconstruct = train_images[n:]

    random.seed(5)
    reconstruction_indices = random.sample(range(train_images.shape[0]), n)
    images_to_reconstruct = train_images[reconstruction_indices]

    # reconstruct the original vectors
    reconstructed_images = vae.decoder.predict(z_mean[reconstruction_indices])

    # create figure to hold subplots
    fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))

    # plot each subplot
    for i in range(0, n - 1):
        # normalise the images
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

    plt.savefig("Variational Eagle/Reconstructions/Training/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    plt.show()
    plt.close()


    if n_flows > 0:

        # reconstruct the transformed vectors images
        reconstructed_images = vae.decoder.predict(z_transformed[reconstruction_indices])

        # create figure to hold subplots
        fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))

        # plot each subplot
        for i in range(0, n - 1):
            # normalise the images
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

        plt.savefig("Variational Eagle/Reconstructions/Training/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed")
        plt.show()
        plt.close()





    # # testing reconstructions
    #
    # # create a subset of the testing data to reconstruct (first n images)
    # images_to_reconstruct = test_images[n:]
    #
    # # generate the extracted features for these images
    # z_mean, _, _, _ = vae.encoder.predict(images_to_reconstruct)
    #
    # # get the sampling layer
    # sampling_layer = None
    # for layer in encoder.layers:
    #     if isinstance(layer, Sampling):
    #         sampling_layer = layer
    #         break
    #
    # # get the flows from the sampling layer
    # flows = sampling_layer.flows
    #
    # # transform the mean vectors
    # z_transformed, _ = apply_flows(z_mean, flows)
    #
    # # reconstruct the original vectors
    # reconstructed_images = vae.decoder.predict(z_mean)
    #
    # # create figure to hold subplots
    # fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))
    #
    # # plot each subplot
    # for i in range(0, n - 1):
    #     # normalise the images
    #     original_image = normalise_independently(images_to_reconstruct[i])
    #     reconstructed_image = normalise_independently(reconstructed_images[i])
    #
    #     # show the original image (remove axes)
    #     axs[0, i].imshow(original_image)
    #     axs[0, i].get_xaxis().set_visible(False)
    #     axs[0, i].get_yaxis().set_visible(False)
    #
    #     # show the reconstructed image (remove axes)
    #     axs[1, i].imshow(reconstructed_image)
    #     axs[1, i].get_xaxis().set_visible(False)
    #     axs[1, i].get_yaxis().set_visible(False)
    #
    # plt.savefig("Variational Eagle/Reconstructions/Testing/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    # plt.show()
    #
    # # reconstruct the transformed vectors images
    # reconstructed_images = vae.decoder.predict(z_transformed)
    #
    # # create figure to hold subplots
    # fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))
    #
    # # plot each subplot
    # for i in range(0, n - 1):
    #     # normalise the images
    #     original_image = normalise_independently(images_to_reconstruct[i])
    #     reconstructed_image = normalise_independently(reconstructed_images[i])
    #
    #     # show the original image (remove axes)
    #     axs[0, i].imshow(original_image)
    #     axs[0, i].get_xaxis().set_visible(False)
    #     axs[0, i].get_yaxis().set_visible(False)
    #
    #     # show the reconstructed image (remove axes)
    #     axs[1, i].imshow(reconstructed_image)
    #     axs[1, i].get_xaxis().set_visible(False)
    #     axs[1, i].get_yaxis().set_visible(False)
    #
    # plt.savefig("Variational Eagle/Reconstructions/Testing/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed")
    # plt.show()









    # loss plot for run

    fig, axs = plt.subplots(3, 3, figsize=(30, 15))

    axs[0][0].plot(range(1, epochs+1), model_loss.history["loss"], label="Total Loss", color="black")
    axs[0][0].plot(range(1, epochs+1), model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    axs[0][0].plot(range(1, epochs+1), model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    axs[0][0].legend()
    axs[0][0].set_xlabel("Epoch")
    axs[0][0].set_ylabel("Loss")

    axs[0][1].plot(range(1, 11), model_loss.history["loss"][:10], label="Total Loss", color="black")
    axs[0][1].plot(range(1, 11), model_loss.history["reconstruction_loss"][:10], label="Reconstruction Loss", color="C0")
    axs[0][1].plot(range(1, 11), model_loss.history["kl_loss"][:10], label="KL Divergence", color="C1")
    axs[0][1].legend()
    axs[0][1].set_xlabel("Epoch")
    axs[0][1].set_ylabel("Loss")

    axs[0][2].plot(range(200, epochs+1), model_loss.history["loss"][199:], label="Total Loss", color="black")
    axs[0][2].plot(range(200, epochs+1), model_loss.history["reconstruction_loss"][199:], label="Reconstruction Loss", color="C0")
    axs[0][2].plot(range(200, epochs+1), model_loss.history["kl_loss"][199:], label="KL Divergence", color="C1")
    axs[0][2].legend()
    axs[0][2].set_xlabel("Epoch")
    axs[0][2].set_ylabel("Loss")




    axs[1][0].plot(range(1, epochs+1), model_loss.history["reconstruction_loss"], label="Reconstruction Loss", color="C0")
    axs2 = axs[1][0].twinx()
    axs2.plot(range(1, epochs+1), model_loss.history["kl_loss"], label="KL Divergence", color="C1")
    lines = axs[1][0].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    labels = axs[1][0].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    axs[1][0].legend(lines, labels)
    # axs[1].legend()
    axs[1][0].set_xlabel("Epoch")
    axs[1][0].set_ylabel("Reconstruction Loss")
    axs2.set_ylabel("KL Divergence")

    axs[1][1].plot(range(1, 11), model_loss.history["reconstruction_loss"][:10], label="Reconstruction Loss", color="C0")
    axs2 = axs[1][1].twinx()
    axs2.plot(range(1, 11), model_loss.history["kl_loss"][:10], label="KL Divergence", color="C1")
    lines = axs[1][1].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    labels = axs[1][1].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    axs[1][1].legend(lines, labels)
    # axs[1].legend()
    axs[1][1].set_xlabel("Epoch")
    axs[1][1].set_ylabel("Reconstruction Loss")
    axs2.set_ylabel("KL Divergence")

    axs[1][2].plot(range(200, epochs+1), model_loss.history["reconstruction_loss"][199:], label="Reconstruction Loss", color="C0")
    axs2 = axs[1][2].twinx()
    axs2.plot(range(200, epochs+1), model_loss.history["kl_loss"][199:], label="KL Divergence", color="C1")
    lines = axs[1][2].get_legend_handles_labels()[0] + axs2.get_legend_handles_labels()[0]
    labels = axs[1][2].get_legend_handles_labels()[1] + axs2.get_legend_handles_labels()[1]
    axs[1][2].legend(lines, labels)
    # axs[1].legend()
    axs[1][2].set_xlabel("Epoch")
    axs[1][2].set_ylabel("Reconstruction Loss")
    axs2.set_ylabel("KL Divergence")



    # plt.savefig("Variational Eagle/Loss Plots/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epochs_" + str(batch_size) + "_bs_loss_" + str(run))
    plt.savefig("Variational Eagle/Loss Plots/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run))
    plt.show()
    plt.close()





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







    #
    # # Training Reconstructions
    #
    # # number of images to reconstruct
    # n = 12
    #
    # # create a subset of the validation data to reconstruct (first 10 images)
    # images_to_reconstruct = train_images[n:]
    #
    # # reconstruct the images
    # test_features, _, _, _ = vae.encoder.predict(images_to_reconstruct)
    # reconstructed_images = vae.decoder.predict(test_features)
    #
    # # create figure to hold subplots
    # fig, axs = plt.subplots(2, n - 1, figsize=(18, 5))
    #
    # # plot each subplot
    # for i in range(0, n - 1):
    #     original_image = normalise_independently(images_to_reconstruct[i])
    #     reconstructed_image = normalise_independently(reconstructed_images[i])
    #
    #     # show the original image (remove axes)
    #     axs[0, i].imshow(original_image)
    #     axs[0, i].get_xaxis().set_visible(False)
    #     axs[0, i].get_yaxis().set_visible(False)
    #
    #     # show the reconstructed image (remove axes)
    #     axs[1, i].imshow(reconstructed_image)
    #     axs[1, i].get_xaxis().set_visible(False)
    #     axs[1, i].get_yaxis().set_visible(False)
    #
    # # plt.savefig("Variational Eagle/Reconstructions/Training/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_reconstruction_" + str(run))
    # plt.savefig("Variational Eagle/Reconstructions/Training/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    # plt.show()
    #
    #
    #
    #
    #
    #
    #
    # # Testing Reconstructions
    #
    # # number of images to reconstruct
    # n = 12
    #
    # # create a subset of the validation data to reconstruct (first 10 images)
    # images_to_reconstruct = test_images[:n]
    # # images_to_reconstruct = train_images[n:]
    #
    # # reconstruct the images
    # test_features, _, _, _ = vae.encoder.predict(images_to_reconstruct)
    # reconstructed_images = vae.decoder.predict(test_features)
    #
    # # create figure to hold subplots
    # fig, axs = plt.subplots(2, n-1, figsize=(18,5))
    #
    # # plot each subplot
    # for i in range(0, n-1):
    #
    #     original_image = normalise_independently(images_to_reconstruct[i])
    #     reconstructed_image = normalise_independently(reconstructed_images[i])
    #
    #     # show the original image (remove axes)
    #     axs[0,i].imshow(original_image)
    #     axs[0,i].get_xaxis().set_visible(False)
    #     axs[0,i].get_yaxis().set_visible(False)
    #
    #     # show the reconstructed image (remove axes)
    #     axs[1,i].imshow(reconstructed_image)
    #     axs[1,i].get_xaxis().set_visible(False)
    #     axs[1,i].get_yaxis().set_visible(False)
    #
    # # plt.savefig("Variational Eagle/Reconstructions/Testing/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_reconstruction_" + str(run))
    # plt.savefig("Variational Eagle/Reconstructions/Testing/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default")
    # plt.show()










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


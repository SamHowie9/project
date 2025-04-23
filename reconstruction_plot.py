import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random





encoding_dim = 25
run = 1

# select which gpu to use
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# number of epochs for run
epochs = 750

# batch size for run
batch_size = 32






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
# bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
bad_fit = all_properties[((all_properties["flag_r"] == 1) |
                          (all_properties["flag_r"] == 4) |
                          (all_properties["flag_r"] == 5) |
                          (all_properties["flag_r"] == 6))].index.tolist()

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















fig, axs = plt.subplots(8, 12, figsize=(12, 8))
# fig, axs = plt.subplots(3, 12, figsize=(12, 3))

images_to_reconstruct = test_images[:12]

# plot each subplot
for i in range(0, 12):

    original_image = normalise_independently(images_to_reconstruct[i])

    # show the original image (remove axes)
    axs[0, i].imshow(original_image)
    # axs[0, i].get_xaxis().set_visible(False)
    # axs[0, i].get_yaxis().set_visible(False)

    axs[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

axs[0][0].set_ylabel("Original")





for run_number, (beta, filename) in enumerate([[0.01, "01"], [0.001, "001"], [0.0001, "0001"], [0.00001, "00001"], [0.000001, "000001"], [0.0000001, "0000001"], [0.00000001, "00000001"]]):
# for run_number, (beta, filename) in enumerate([[0.0001, "0001"], [0.00001, "00001"]]):


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

                # reconstruction loss
                # reconstruction_loss = ops.sum(ops.mean(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
                reconstruction_loss = ops.mean(keras.losses.binary_crossentropy(data, reconstruction))

                # kl loss
                kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                kl_loss = ops.mean(kl_loss)
                # kl_loss = np.tanh(kl_loss)

                # total loss
                # total_loss = reconstruction_loss + kl_loss
                # total_loss = reconstruction_loss + (0.0000001 * kl_loss)
                total_loss = reconstruction_loss + (beta * kl_loss)


                print("Reconstruction Loss Shape:", reconstruction_loss.shape)
                print("KL Loss Shape:", kl_loss.shape)
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





    vae.load_weights("Variational Eagle/Weights/Test/bce_beta_" + filename + ".weights.h5")







    # number of images to reconstruct
    n = 12

    # create a subset of the validation data to reconstruct (first 10 images)
    images_to_reconstruct = test_images[:n]
    # images_to_reconstruct = train_images[n:]

    # reconstruct the images
    test_features, _, _ = vae.encoder.predict(images_to_reconstruct)
    reconstructed_images = vae.decoder.predict(test_features)

    # plot each subplot
    for i in range(0, n):

        reconstructed_image = normalise_independently(reconstructed_images[i])

        # show the reconstructed image (remove axes)
        axs[run_number+1, i].imshow(reconstructed_image)
        # axs[run_number+1, i].get_xaxis().set_visible(False)
        # axs[run_number+1, i].get_yaxis().set_visible(False)

        axs[run_number+1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    axs[run_number+1][0].set_ylabel(str(beta))

# plt.savefig("Variational Eagle/Reconstructions/Testing/fully_balanced_mean_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_reconstruction_" + str(run))
plt.savefig("Variational Eagle/Plots/beta_comparison_reconstruction_2", bbox_inches='tight')
plt.show()


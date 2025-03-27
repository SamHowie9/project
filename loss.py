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
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



plt.style.use("default")
sns.set_style("ticks")





encoding_dim = 10
run = 1
epochs = 750
batch_size = 32





# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T



# load the images as a fully balanced dataset

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
# train_images = all_images[:-200]
# test_images = np.array(all_images[-200:])
#
#
# # load the filenames of the augmented elliptical images
# augmented_galaxies =  os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/")
#
# for galaxy in augmented_galaxies:
#
#     # load each augmented image
#     image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/" + galaxy)
#
#     # normalise the image
#     image = normalise_independently(image)
#
#     # add the image to the training set (not the testing set)
#     train_images.append(image)
#
#
#
# # load the filenames of the augmented unknown images
# augmented_galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/")
#
# print("...")
#
# for galaxy in augmented_galaxies:
#
#     # load each augmented image
#     image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/" + galaxy)
#
#     # normalise the image
#     image = normalise_independently(image)
#
#     # add the image to the training set (not the testing set)
#     train_images.append(image)
#
# # convert the training set to a numpy array
# train_images = np.array(train_images)







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
# print(len(all_images))
# print(train_images.shape)
# print(test_images.shape)









# load only the elliptical galaxies

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


# take only the sprial galaxies
all_properties = all_properties[all_properties["n_r"] >= 4]

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


# split the data into training and testing data (12 images used for testing)
train_images = np.array(all_images[:-12])
test_images = np.array(all_images[-12:])








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

        with tf.GradientTape() as tape:

            # get the latent representation (run image through the encoder)
            z_mean, z_log_var, z = self.encoder(data)

            # form the reconstruction (run latent representation through decoder)
            reconstruction = self.decoder(z)

            # calculate the binary cross entropy reconstruction loss (sum over each pixel and average (mean) across each channel and across the batch)
            # reconstruction_loss = ops.mean(
            #     ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2),
            #     )
            # )
            reconstruction_loss = ops.mean(keras.losses.binary_crossentropy(data, reconstruction))

            # calculate the kl divergence (sum over each latent feature and average (mean) across the batch)
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            # kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            kl_loss = ops.mean(kl_loss)

            # total loss is the sum of reconstruction loss and kl divergence
            total_loss = reconstruction_loss + kl_loss

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

        # get the latent distribution
        z_mean, z_log_var = inputs

        # find the batch size and number of latent features (dim)
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]

        # generate the random variables
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)

        # perform reparameterization trick
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon



for run in [1, 2, 3]:

    total_loss_all = []
    reconstruction_loss_all = []
    kl_loss_all = []

    # total_loss_all = list(np.load("Variational Eagle/Loss/Fully Balanced/total_loss_" + str(run) + ".npy"))
    # reconstruction_loss_all = list(np.load("Variational Eagle/Loss/Fully Balanced/reconstruction_loss_" + str(run) + ".npy"))
    # kl_loss_all = list(np.load("Variational Eagle/Loss/Fully Balanced/kl_loss_" + str(run) + ".npy"))



    for encoding_dim in range(1, 21):


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


        # # load the weights
        vae.load_weights("Variational Eagle/Weights/Elliptical/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5")



        # get the latent representations (run image through the encoder)
        z_mean, z_log_var, z = vae.encoder.predict(train_images)

        print(z_mean.shape)

        # reconstruct the image
        reconstructed_images = vae.decoder.predict(z_mean)

        print(reconstructed_images.shape)

        # get the reconstruction loss
        reconstruction_loss = ops.mean(keras.losses.binary_crossentropy(train_images, reconstructed_images)).numpy().item()

        # calculate the kl divergence (sum over each latent feature and average (mean) across the batch)
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        # kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        kl_loss = ops.mean(kl_loss).numpy().item()

        total_loss_all.append(reconstruction_loss + kl_loss)
        reconstruction_loss_all.append(reconstruction_loss)
        kl_loss_all.append(kl_loss)



    np.save("Variational Eagle/Loss/Elliptical/total_loss_" + str(run), total_loss_all)
    np.save("Variational Eagle/Loss/Elliptical/reconstruction_loss_" + str(run), reconstruction_loss_all)
    np.save("Variational Eagle/Loss/Elliptical/kl_loss_" + str(run), kl_loss_all)






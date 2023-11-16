import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
import tensorflow as tf
import tensorflow-probability as tfp
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# select to use GPU 0 on cosma
os.environ["CUDA_VISIBLE_DEVICES"]="0" # for GPU


# returns a numpy array of the images to train the model
def get_images():

    # stores an empty list to contain all the image data to train the model
    all_images = []

    # loop through the directory containing all the image files
    for file in os.listdir("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/"):

        # open the fits file and get the image data (this is a numpy array of each pixel value)
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/" + file)

        # append the image data to the main list containing all data of all the images
        all_images.append(image)

    # return this list
    return all_images


# get the images and labels to train the model
all_images = get_images()

# # find the number of images that you will test the model on
# testing_count = int(len(all_images)/4)
#
# # split the data into training and testing data based on this number (and convert from list to numpy array of shape (256,256,3) given it is an rgb image
# # train_images = np.array(all_images[:testing_count*3])
# # test_images = np.array(all_images[testing_count:])



# Define keras tensor for the encoder
input_image_encoder = keras.Input(shape=(256, 256, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image_encoder)    # (128, 128, 32)
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                      # (64, 64, 64)
x = Flatten()(x)                                                                                            # (262144) = (64 * 64 * 64)
z_mean = Dense(units=2, activation="relu", name="z_mean")(x)                                                # (2)
z_log_var = Dense(units=2, activation="relu", name="z_log_var")(x)                                          # (2)
z = Sampling()([z_mean, z_log_var])

# build the encoder
encoder = keras.Model(input_image_encoder, [z_mean, z_log_var, z], name="encoder")


# Define keras tensor for the decoder
input_image_decoder = keras.Input(shape=(2))                                                                # (2)

# layers for the decoder
x = Dense(units=64*64*32, activation="relu")(input_image_decoder)                                           # (131072) = (64 * 64 * 32)
x = Reshape((64, 64, 32))(x)                                                                                # (64, 64, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (128, 128, 64)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (265, 256, 32)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="relu", padding="same")(x)                   # (256, 256, 3)

# build the decoder
decoder = keras.Model(input_image, decoded, name="decoder")



# build the VAE Model with a custom train step
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super()__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    # custom train step
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# train the VAE
vae = VAE(encoder, decoder)
vae.compile(optimiser="adam")
vae.fit(all_images, epochs=3, batch_size=1)



# plotting latent space
def plot_latent_space(model, n=10):

    image_size = 32
    scale = 1

    # create an image matrix
    figure = np.zeros((image_size * n, image_size * n))

    # linearly spaced coordinates corresponding to 2d plot
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerategrid_x:
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decoder.predict(z_sample)
            image = x_decoded[0].reshape(image_size, image_size)

            figure[
                i * image_size : (i+1) * image_size,
                j * image_size : (j+1) * image_size,
            ] = image

    plt.figure(figsize = (20, 20))
    start_range = image_size // 2
    end_range = start_range + (n * image_size)
    pixel_range = np.arrange(start_range, end_range, image_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")


plot_latent_space(vae)


plt.show()
plt.savefig("latent_space")
import os

from matplotlib.pyplot import figure

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg




# number of extracted features
encoding_dim = 20

# number of epochs for run
epochs = 300






# Define VAE model with custom time step
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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss + kl_loss))
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




# define sampling layer
class Sampling(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon





# encoding_dim = 20




# Define keras tensor for the encoder
input_image = keras.Input(shape=(256, 256, 3))                                                                  # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)                           # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)                           # (8, 8, 4)
x = Flatten()(x)                                                                                                # (256)
x = Dense(units=64)(x)                                                                                          # (64)
z_mean = Dense(encoding_dim, name="z_mean")(x)
z_log_var = Dense(encoding_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

# build the encoder
encoder = keras.Model(input_image, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# Define keras tensor for the decoder
latent_input = keras.Input(shape=(encoding_dim,))

# layers for the decoder
x = Dense(units=64)(latent_input)                                                                               # (64)
x = Dense(units=256)(x)                                                                                         # (256)
x = Reshape((8, 8, 4))(x)                                                                                       # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)                  # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)                  # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (128, 128, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)    # (128, 128, 3) (forces output between 0 and 1)
# decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="relu", padding="same", name="decoded")(x)     # (128, 128, 3) (forces output to be positive)

# build the decoder
decoder = keras.Model(latent_input, decoded, name="decoder")
decoder.summary()




# build and compile the VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# load the weights
vae.load_weights("Variational Eagle/Weights/Normalised to g/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_1.weights.h5")





# number of varying features
num_varying_features = 15

# chosen extracted feature to vary
chosen_feature = 19
chosen_feature_2 = 12

# chosen pca feature to vary
chosen_pca_feature = 1




# load the extracted features
extracted_features = np.load("Variational Eagle/Extracted Features/Normalised to g/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_features_1.npy")[0]

# list of medians for all extracted features
med_extracted_features = [np.median(extracted_features.T[i]) for i in range(encoding_dim)]

# values to vary the chosen extracted feature (equally spaced values between min and max)
# varying_feature_values = np.linspace(np.min(extracted_features.T[chosen_feature]), np.max(extracted_features.T[chosen_feature]), num_varying_features)
varying_feature_values = [-4, -3] + list(np.linspace(-2, 3, 13))
# varying_feature_values = [-5, -3] + list(np.linspace(-2.5, 2, 12)) + [2.5]

# second feature
# varying_feature_values_2 = np.linspace(np.min(extracted_features.T[chosen_feature_2]), np.max(extracted_features.T[chosen_feature_2]), num_varying_features)
varying_feature_values_2 = [-0.3, -0.2] + list(np.linspace(-0.15, 0.05, 12)) + [0.1]
# varying_feature_values_2 = [-0.1, -0.8] + list(np.linspace(-0.075, 0.1, 12)) + [0.15]



# apply pca on the extracted features and project the extracted features
pca = PCA(n_components=11).fit(extracted_features)
pca_features = pca.transform(extracted_features)

# list of medians for all pca features (equally spaced values between min and max)
med_pca_features = [np.median(pca_features.T[i]) for i in range(11)]

# values to vary the chosen pca feature
# varying_pca_feature_values = np.linspace(np.min(pca_features.T[chosen_pca_feature]), np.max(pca_features.T[chosen_pca_feature]), num_varying_features)
varying_pca_feature_values = [-4, -3] + list(np.linspace(-2, 2, 13))


fig, axs = plt.subplots(5, num_varying_features, figsize=(15, 6))

for i in range(num_varying_features):

    # 8, 10
    temp_features = med_extracted_features.copy()
    temp_features[chosen_feature] = varying_feature_values[i]
    temp_features = np.expand_dims(temp_features, axis=0)

    reconstruction = vae.decoder.predict(temp_features)[0]

    axs[0, i].imshow(reconstruction)
    axs[0, i].get_xaxis().set_visible(False)
    axs[0, i].get_yaxis().set_visible(False)



    temp_features_2 = med_extracted_features.copy()
    temp_features_2[chosen_feature_2] = varying_feature_values_2[i]
    temp_features_2 = np.expand_dims(temp_features_2, axis=0)

    reconstruction_2 = vae.decoder.predict(temp_features_2)[0]

    axs[1, i].imshow(reconstruction_2)
    axs[1, i].get_xaxis().set_visible(False)
    axs[1, i].get_yaxis().set_visible(False)



    temp_features_3 = med_extracted_features.copy()
    temp_features_3[chosen_feature] = varying_feature_values[i]
    temp_features_3[chosen_feature_2] = varying_feature_values_2[i]
    temp_features_3 = np.expand_dims(temp_features_3, axis=0)

    reconstruction_3 = vae.decoder.predict(temp_features_3)[0]

    axs[2, i].imshow(reconstruction_3)
    axs[2, i].get_xaxis().set_visible(False)
    axs[2, i].get_yaxis().set_visible(False)



    fig.delaxes(axs[3, i])


    # 1
    temp_pca_features = med_pca_features.copy()
    temp_pca_features[chosen_pca_feature] = varying_pca_feature_values[i]
    temp_pca_features = pca.inverse_transform(temp_pca_features)
    temp_pca_features = np.expand_dims(temp_pca_features, axis=0)

    reconstruction = vae.decoder.predict(temp_pca_features)[0]

    axs[4, i].imshow(reconstruction)
    axs[4, i].get_xaxis().set_visible(False)
    axs[4, i].get_yaxis().set_visible(False)



axs[0,2].set_title("Varying VAE Feature 19                              ")
axs[1,2].set_title("Varying VAE Feature 12                              ")
axs[2,2].set_title("Varying VAE Feature 19 and 12                       ")
axs[4,2].set_title("Varying PCA Feature 1                               ")


plt.savefig("Variational Eagle/Plots/transition_plot_g_normalised_vae_vs_pca", bbox_inches='tight')
plt.show()




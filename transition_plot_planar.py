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
from sklearn.decomposition import PCA
from skimage import color





run = 16
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32


os.environ["CUDA_VISIBLE_DEVICES"]="0"



# scale font on plots
default_size = plt.rcParams['font.size']
plt.rcParams.update({'font.size': default_size * 6})






# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T





# for run in range(1, 11):
for run in [run]:

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

            # perform reparameterization trick
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
        z = tf.clip_by_value(z, -4+1e-4, 4-1e-4)

        # apply the flows
        for flow in flows:
            z, _ = flow(z)

        # return the transformed vector
        return z





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





    vae.build(input_shape=(None, 256, 256, 3))

    # or load the weights from a previous run
    vae.load_weights("Variational Eagle/Weights/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")
    # vae.load_weights("Variational Eagle/Weights/Ellipticals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")



    # load the original and transformed features
    z_mean = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    z_transformed = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
    # z_mean = np.load("Variational Eagle/Extracted Features/Ellipticals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    # z_transformed = np.load("Variational Eagle/Extracted Features/Ellipticals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

    # perform PCA on both sets of features
    # pca_mean = PCA(n_components=0.999, svd_solver="full").fit(z_mean)
    # z_mean = pca_mean.transform(z_mean)
    # pca_transformed = PCA(n_components=0.999, svd_solver="full").fit(z_transformed)
    # z_transformed = pca_transformed.transform(z_transformed)
    # pca = pca_transformed



    # select transformed or mean
    extracted_features = z_mean
    # extracted_features = z_transformed







    # all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")
    #
    # # all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_spirals.csv")
    # # all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_ellipticals.csv")
    # # all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_transitional.csv")

    # spirals only
    # spiral_indices = all_properties[all_properties["DiscToTotal"] > 0.2].index.tolist()
    # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"] > 0.2]

    # ellipticals only
    # spiral_indices = all_properties[all_properties["DiscToTotal"] < 0.1].index.tolist()
    # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"] < 0.1]

    # transitional
    # spiral_indices = all_properties[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")].index.tolist()
    # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")]

    # original images
    # all_properties = all_properties_real
    # extracted_features = extracted_features[:len(all_properties)]










    # transition plot for all extracted features

    num_varying_features = 13

    med_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]
    print(len(med_features))

    fig, axs = plt.subplots(len(extracted_features.T), num_varying_features, figsize=(num_varying_features*5, len(extracted_features.T)*5), dpi=100)

    # loop through each feature
    for i in range(len(extracted_features.T)):

        # get the incremental values for that feature
        varying_feature_values = np.linspace(np.min(extracted_features.T[i]), np.max(extracted_features.T[i]), num_varying_features)


        for j in range(num_varying_features):

            temp_features = med_features.copy()
            temp_features[i] = varying_feature_values[j]

            # temp_features = pca.inverse_transform(temp_features)

            temp_features = np.expand_dims(temp_features, axis=0)

            reconstruction = vae.decoder.predict(temp_features)[0]

            axs[i][j].imshow(reconstruction)
            axs[i][j].set_aspect("auto")
            axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

            # reconstruction_grey = color.rgb2gray(reconstruction)
            # axs[i][j].imshow(reconstruction_grey, cmap="gray_r")
            # axs[i][j].set_aspect("auto")
            # axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)


            # remove the spines
            # for spine in axs[i][j].spines.values():
            #     spine.set_visible(False)

        axs[i][0].set_ylabel(i+1, rotation=0, labelpad=40, va='center')

    fig.text(0.09, 0.5, 'Extracted Features', va='center', rotation='vertical')


    fig.subplots_adjust(wspace=0, hspace=0.05)

    plt.savefig("Variational Eagle/Transition Plots/Normalising Flow Balanced/latent_" + str(encoding_dim) + "_flows_" + str(n_flows) + "_" + str(run) + "_balanced", bbox_inches='tight')
    # plt.savefig("Variational Eagle/Transition Plots/Normalising Flow Balanced/pca_latent_" + str(encoding_dim) + "_flows_" + str(n_flows) + "_" + str(run) + "_balanced.pdf", bbox_inches='tight')
    # plt.savefig("Variational Eagle/Transition Plots/Ellipticals/latent_" + str(encoding_dim) + "_flows_" + str(n_flows) + "_" + str(run) + "_balanced", bbox_inches='tight')
    # plt.savefig("Variational Eagle/Transition Plots/Spirals/pca_latent_" + str(encoding_dim) + "_flows_" + str(n_flows) + "_" + str(run) + "_balanced.pdf", bbox_inches='tight')

    plt.show(block=False)
    plt.close()






    # transition plot for group of features

    # num_varying_features = 13
    #
    # med_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]
    # print(len(med_features))
    #
    # chosen_features = [18, 13, 6]
    #
    # fig, axs = plt.subplots(len(chosen_features), num_varying_features, figsize=(num_varying_features*2.5, len(chosen_features)*2.5), dpi=100)
    #
    # for i, feature in enumerate(chosen_features):
    #
    #     varying_feature_values = np.linspace(np.min(extracted_features.T[feature-1]), np.max(extracted_features.T[feature-1]), num_varying_features)
    #
    #     for j in range(num_varying_features):
    #
    #         temp_features = med_features.copy()
    #         temp_features[feature-1] = varying_feature_values[j]
    #
    #         temp_features = np.expand_dims(temp_features, axis=0)
    #
    #         # temp_features = pca.inverse_transform(temp_features)
    #         # temp_features = np.expand_dims(temp_features, axis=0)
    #
    #         reconstruction = vae.decoder.predict(temp_features)[0]
    #
    #         axs[i][j].imshow(reconstruction)
    #         axs[i][j].set_aspect("auto")
    #         axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    #
    #         # axs[i][j].set_xlabel(round(varying_feature_values[j], 2))
    #         #
    #         # if j == (num_varying_features - 1)/2:
    #         #     axs[i][j].set_xlabel(str(round(varying_feature_values[j], 2)) + "\nPCA Feature " + str(feature))
    #
    #     # axs[i][0].set_ylabel(feature+1, rotation=0, labelpad=40, va='center')
    #
    # axs[0][0].set_ylabel("dCor < 0.2", fontsize=60, rotation=0, labelpad=15, va='center', ha="right")
    # axs[1][0].set_ylabel("Sérsic Index\ndCor = 0.74", fontsize=60, rotation=0, labelpad=15, va="center", ha="right")
    # axs[2][0].set_ylabel("Half-Light Radius\ndCor = 0.81", fontsize=60, rotation=0, labelpad=15, va="center", ha="right")
    #
    #
    # fig.subplots_adjust(wspace=0, hspace=0.05)
    #
    # plt.savefig("Variational Eagle/Transition Plots/Normalising Flow Balanced/selected_latent_feature_transitions", bbox_inches='tight')
    # plt.savefig("Variational Eagle/Transition Plots/Normalising Flow Balanced/selected_latent_feature_transitions.pdf", bbox_inches='tight')
    # # plt.savefig("Variational Eagle/Transition Plots/Spirals/selected_latent_feature_transitions", bbox_inches='tight')
    # # plt.savefig("Variational Eagle/Transition Plots/Spirals/selected_latent_feature_transitions.pdf", bbox_inches='tight')
    #
    # plt.show(block=False)
    # plt.close()









    # transition plot residual

    # med_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]
    #
    # max_corr = np.load("Variational Eagle/Correlation Plots/Normalising Flows Balanced/Normal/max_corr.npy")
    #
    # fig, axs = plt.subplots(len(extracted_features.T), 3, figsize=(12, len(extracted_features.T)*2.5), dpi=100)
    #
    # # for i, feature in enumerate(chosen_features):
    # for i in range(len(extracted_features.T)):
    #
    #     min_features = med_features.copy()
    #     min_features[i] = np.min(extracted_features.T[i])
    #     min_features = np.expand_dims(min_features, axis=0)
    #     min_reconstruction = vae.decoder.predict(min_features)[0]
    #
    #     max_features = med_features.copy()
    #     max_features[i] = np.max(extracted_features.T[i])
    #     max_features = np.expand_dims(max_features, axis=0)
    #     max_reconstruction = vae.decoder.predict(max_features)[0]
    #
    #     residual = abs(max_reconstruction - min_reconstruction)
    #     residual = color.rgb2gray(residual)
    #
    #
    #     axs[i][0].imshow(min_reconstruction)
    #     axs[i][0].set_aspect("equal")
    #     axs[i][0].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    #     axs[i][0].set_ylabel(i+1, rotation=0, fontsize=45, labelpad=10, va='center', ha="right")
    #
    #     axs[i][1].imshow(max_reconstruction)
    #     axs[i][1].set_aspect("equal")
    #     axs[i][1].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    #
    #     axs[i][2].imshow(residual, cmap="gray_r")
    #     axs[i][2].set_aspect("equal")
    #     axs[i][2].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    #
    #     avg_residual = round(np.mean(residual), 5)
    #     std = round(np.std(extracted_features.T[i]), 5)
    #     dCor = round(max_corr[i], 5)
    #
    #     axs[i][2].yaxis.set_label_position("right")  # ← must be separate
    #     axs[i][2].set_ylabel(("res = " + f"{avg_residual:.5f}" + "\nstd = " + str(std) + "\ndCor = " + str(dCor)), rotation=0, fontsize=45, labelpad=10, va='center', ha="left")
    #
    # fig.text(0.0, 0.5, 'Extracted Features', va='center', rotation='vertical')
    # axs[0][0].set_title("Minimum", fontsize=50)
    # axs[0][1].set_title("Maximum", fontsize=50)
    # axs[0][2].set_title("Residual", fontsize=50)
    #
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)
    #
    # plt.savefig("Variational Eagle/Transition Plots/Normalising Flow Balanced/latent_transition_residual_" + str(run), bbox_inches="tight")
    # # plt.savefig("Variational Eagle/Transition Plots/Spirals/latent_transition_residual_" + str(run), bbox_inches="tight")
    # plt.show(block=False)
    # plt.close()





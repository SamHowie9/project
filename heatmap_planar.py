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
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from skimage import io, color





tf.keras.mixed_precision.set_global_policy('float32')


tfb = tfp.bijectors
tfd = tfp.distributions






run = 3
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32





# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T









chosen_galaxies = np.load("Galaxy Properties/Eagle Properties/chosen_glaxies.npy")

# list to contain all galaxy images
all_images = []

# loop through each galaxy
for i, galaxy in enumerate(chosen_galaxies):

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")

    # normalise the image (each band independently)
    image = normalise_independently(image)

    # add the image to the dataset
    all_images.append(image)

print("Original Dataset", len(all_images))

# split the data into training and testing data (200 images used for testing)
train_images = all_images
# train_images = all_images[:-200]
# test_images = np.array(all_images[-200:])

# print("Training Set", len(train_images))
# print("Testing Set", len(test_images))
# print()



# load the filenames of the augmented elliptical images
augmented_galaxies =  os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")

print("Augmented Ellipticals", len(augmented_galaxies))

for galaxy in augmented_galaxies:

    # load each augmented image
    image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/" + galaxy)

    # normalise the image
    image = normalise_independently(image)

    # add the image to the training set (not the testing set)
    train_images.append(image)



# load the filenames of the augmented transitional images
augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/")

print("Augmented Transitional", len(augmented_galaxies))

for galaxy in augmented_galaxies:

    # load each augmented image
    image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/" + galaxy)

    # normalise the image
    image = normalise_independently(image)

    # add the image to the training set (not the testing set)
    train_images.append(image)

# convert the training set to a numpy array
train_images = np.array(train_images)


print("Training Set", train_images.shape)
# print("Testing Set", test_images.shape)
print()








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
input_image = Input(shape=(256, 256, 3))  # (256, 256, 3)

# layers for the encoder
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)  # (128, 128, 32)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (64, 64, 64)
x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (32, 32, 128)
x = layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (16, 16, 256)
x = layers.Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (8, 8, 512)
# x = Flatten()(x)                                                                                                     # (8*8*512 = 32768)
x = layers.GlobalAveragePooling2D()(x)  # (512)
x = layers.Dense(128, activation="relu")(x)  # (128)

z_mean = layers.Dense(encoding_dim, name="z_mean")(x)
z_log_var = layers.Dense(encoding_dim, name="z_log_var")(x)

z, sum_log_det_jacobians = Sampling(encoding_dim, n_flows=n_flows)([z_mean, z_log_var])

# build the encoder
encoder = Model(input_image, [z_mean, z_log_var, z, sum_log_det_jacobians], name="encoder")
encoder.summary()



# Define keras tensor for the decoder
latent_input = Input(shape=(encoding_dim,))

# layers for the decoder
x = layers.Dense(units=128, activation="relu")(latent_input)  # (64)
x = layers.Dense(units=512, activation="relu")(x)  # (256)
x = layers.Dense(units=8 * 8 * 512, activation="relu")(x)  # (8*8*512 = 32768)
x = layers.Reshape((8, 8, 512))(x)  # (8, 8, 512)
x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (16, 16, 256)
x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (32, 32, 128)
x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (64, 64, 64)
x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)  # (128, 128, 32)
decoded = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="sigmoid", padding="same")(x)  # (256, 256, 3)



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




def normalise_map(x):
    x = tf.math.abs(x)
    x -= tf.reduce_min(x)
    x /= tf.reduce_max(x) + 1e-8
    return x



def latent_flows_saliency(encoder, image, flows=False, feature=None, smoothing_sigma=None):

    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    image = tf.Variable(image)

    with tf.GradientTape() as tape:

        tape.watch(image)
        z_mean, z_log_var, z, _ = encoder(image)

        if flows:
            # get the sampling layer
            sampling_layer = None
            for layer in encoder.layers:
                if isinstance(layer, Sampling):
                    sampling_layer = layer
                    break

            # get the flows from the sampling layer
            flows = sampling_layer.flows

            # transform the mean vectors
            z_transformed, _ = apply_flows(z_mean, flows)
        else:
            z_transformed = z_mean



        if feature is not None:
            target = z_transformed[:, feature]
        else:
            target = z_transformed

        # target = z[:, feature] if use_flow_output else z_mean[:, feature]



    # gradient of feature wrt image
    grads = tape.gradient(target, image)
    saliency = tf.reduce_sum(grads**2, axis=-1)[0]


    saliency = normalise_map(saliency)

    if smoothing_sigma is not None:
        saliency = tf.numpy_function(
            lambda m: tf.squeeze(
                gaussian_filter(m[None, ..., None], sigma=smoothing_sigma)),
            [saliency], tf.float32)

    saliency = normalise_map(saliency)

    return saliency.numpy()







def latent_saliency(encoder, image, feature=None, smoothing_sigma=None):

    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    image = tf.Variable(image)

    with tf.GradientTape() as tape:

        z_mean, _, _, _ = encoder(image)
        z_mean = z_mean[0]


        if feature is not None:
            latent_feature = z_mean[feature]
        else:
            latent_feature = z_mean

    grads = tape.gradient(latent_feature, image)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]

    # normalise saliency
    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8

    if smoothing_sigma is not None:

        # apply gaussian filter
        saliency = gaussian_filter(saliency, sigma=smoothing_sigma)

        # normalise saliency again
        saliency -= saliency.min()
        saliency /= saliency.max() + 1e-8

    return saliency










def pca_saliency(encoder, image, pca_components, pca_component_index, smoothing_sigma=None):

    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    image = tf.Variable(image)

    with tf.GradientTape() as tape:

        # tape.watch(image)
        z_mean, _, _, _ = encoder(image)
        z_mean = z_mean[0]

        pca_direction = tf.convert_to_tensor(pca_components[pca_component_index], dtype=tf.float32)
        pca_feature = tf.tensordot(z_mean, pca_direction, axes=1)

    grads = tape.gradient(pca_feature, image)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]

    # normalise saliency
    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8

    if smoothing_sigma is not None:

        # apply gaussian filter
        saliency = gaussian_filter(saliency, sigma=smoothing_sigma)

        # normalise saliency again
        saliency -= saliency.min()
        saliency /= saliency.max() + 1e-8

    return saliency








# scale font on plots
default_size = plt.rcParams['font.size']
plt.rcParams.update({'font.size': default_size * 4})


extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

print(extracted_features.shape)

# pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# pca_components = pca.components_


img_indices = [560, 743, 839, 780, 2785, 2929, 2227, 3382, 495, 437, 2581]




# heatmap for all features

# fig, axs = plt.subplots(extracted_features.shape[1]+1, len(img_indices), figsize=(len(img_indices)*5, extracted_features.shape[1]*5))
#
# for i, img_index in enumerate(img_indices):
#
#     axs[0][i].imshow(train_images[img_index])
#     axs[0][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     for feature in range(0, extracted_features.shape[1]):
#
#         heatmap = latent_saliency(encoder=vae.encoder, image=train_images[img_index], feature=feature, smoothing_sigma=2.0)
#         # heatmap = pca_saliency(encoder=vae.encoder, image=train_images[img_index], pca_components=pca_components, pca_component_index=feature, smoothing_sigma=2.0)
#
#         axs[feature+1][i].imshow(train_images[img_index])
#         axs[feature+1][i].imshow(heatmap, cmap="jet", alpha=0.5)
#         axs[feature+1][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#         axs[feature+1][0].set_ylabel(feature+1, rotation=0, labelpad=40, va='center')
#
#
# plt.savefig("Variational Eagle/Plots/heatmap_individual_smooth", bbox_inches="tight")
# plt.show()






# heatmap for all features

# fig, axs = plt.subplots(2, len(img_indices), figsize=(30, 6))
#
# for i, img_index in enumerate(img_indices):
#
#     axs[0][i].imshow(train_images[img_index])
#     axs[0][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     heatmap = latent_saliency(encoder=vae.encoder, image=train_images[img_index], smoothing_sigma=2.0)
#     # heatmap = pca_saliency(encoder=vae.encoder, image=train_images[img_index], pca_components=pca_components, pca_component_index=feature, smoothing_sigma=2.0)
#
#     axs[1][i].imshow(train_images[img_index])
#     axs[1][i].imshow(heatmap, cmap="jet", alpha=0.5)
#     axs[1][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
# plt.savefig("Variational Eagle/Plots/heatmap_all_smooth", bbox_inches="tight")
# plt.show()








# reconstructions and heatmaps

all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")

reconstruction_indices = [560, 743, 839, 780, 2785, 2929, 2227, 3382, 495, 437, 2581]

extracted_features_reconstruct = extracted_features[reconstruction_indices]
original_images = train_images[reconstruction_indices]


# reconstructions with residual:
fig, axs = plt.subplots(4, len(reconstruction_indices), figsize=(len(reconstruction_indices)*5, 4*5))


# pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
# pca_features = pca.transform(extracted_features_reconstruct)
# pca_features = pca.inverse_transform(pca_features)
# reconstructions = vae.decoder.predict(pca_features)

reconstructions = vae.decoder.predict(extracted_features_reconstruct)

residuals = abs(original_images - reconstructions)

# residuals -= residuals.min()
# residuals /= residuals.max() + 1e-8

for i in range(0, len(reconstruction_indices)):

    original_image = normalise_independently(original_images[i])
    axs[0][i].imshow(original_image)
    axs[0][i].set_aspect("auto")
    axs[0][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    dt = all_properties.loc[reconstruction_indices[i], "DiscToTotal"]
    axs[0][i].set_title("D/T=" + str(round(dt, 3)), fontsize=40)

    axs[1][i].imshow(reconstructions[i])
    axs[1][i].set_aspect("auto")
    axs[1][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)


    # axs[2][i].imshow(residual)
    residual = color.rgb2gray(residuals[i])
    axs[2][i].imshow(residual, cmap="gray_r")
    axs[2][i].set_aspect("auto")
    axs[2][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    avg_residual = str(round(np.mean(residuals[i]), 3))
    axs[2][i].text(0.01, 0.01, avg_residual, transform=axs[2][i].transAxes, ha='left', va='bottom')



    heatmap = latent_saliency(encoder=vae.encoder, image=original_images[i], smoothing_sigma=2.0)
    # heatmap = pca_saliency(encoder=vae.encoder, image=train_images[img_index], pca_components=pca_components, pca_component_index=feature, smoothing_sigma=2.0)

    original_gray = color.rgb2gray(original_image)
    axs[3][i].imshow(original_gray, cmap="gray")
    axs[3][i].imshow(heatmap, cmap="jet", alpha=1)
    axs[3][i].set_aspect("auto")
    axs[3][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)



axs[0][0].set_ylabel("Original")
axs[1][0].set_ylabel("Recons.")
axs[2][0].set_ylabel("Residual")
axs[3][0].set_ylabel("Heatmap")

fig.subplots_adjust(wspace=0.1, hspace=0.05)

plt.savefig("Variational Eagle/Plots/reconstructions_residuals_heatmap_3", bbox_inches="tight")
plt.show()
plt.close()

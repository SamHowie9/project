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



tf.keras.mixed_precision.set_global_policy('float32')


tfb = tfp.bijectors
tfd = tfp.distributions






run = 1
encoding_dim = 30
n_flows = 3
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





# load the images as a balanced dataset
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

        # z = tf.clip_by_value(z, -4+1e-4, 4-1e-4)

        # apply flow transformations
        sum_log_det_jacobian = 0.0
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
        u_hat = self.u + \
                    (tf.nn.softplus(tf.reduce_sum(self.w * self.u)) - 1 - tf.reduce_sum(self.w * self.u)) * self.w / (
                            tf.norm(self.w) ** 2 + 1e-8)

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
vae.load_weights("Variational Eagle/Weights/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")




def normalise_map(x):
    x = tf.math.abs(x)
    x -= tf.reduce_min(x)
    x /= tf.reduce_max(x) + 1e-8
    return x



def latent_saliency(encoder, image, feature=None, smoothing_sigma=None):

    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)  # add batch dim
    image = tf.Variable(image)  # allows in-place grads if you want FGSM later

    with tf.GradientTape() as tape:

        tape.watch(image)
        z_mean, z_log_var, z, _ = encoder(image)

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

        if feature is not None:
            target = z_transformed[:, feature]
        else:
            target = z_transformed

        # target = z[:, feature] if use_flow_output else z_mean[:, feature]



    # ∂ z_i / ∂ x
    grads = tape.gradient(target, image)                    # (1,H,W,3)
    saliency = tf.reduce_sum(grads**2, axis=-1)[0]        # (H,W)   L2 over RGB


    saliency = normalise_map(saliency)

    if smoothing_sigma is not None:
        saliency = tf.numpy_function(
            lambda m: tf.squeeze(
                gaussian_filter(m[None, ..., None], sigma=smoothing_sigma)),
            [saliency], tf.float32)

    saliency = normalise_map(saliency)

    return saliency.numpy()




fig, axs = plt.subplots(encoding_dim+1, 10, figsize=(30, 90))

for img_index in range(0, 10):

    axs[0][img_index].imshow(test_images[img_index])
    axs[0][img_index].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    for feature in range(0, encoding_dim):

        heatmap = latent_saliency(vae.encoder, test_images[img_index], feature, smoothing_sigma=2.0)

        axs[feature+1][img_index].imshow(test_images[img_index])
        axs[feature+1][img_index].imshow(heatmap, cmap="jet", alpha=0.5)
        axs[feature+1][img_index].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)


plt.savefig("Variational Eagle/Plots/heatmap_individual_smooth", bbox_inches="tight")
plt.show()



fig, axs = plt.subplots(2, 10, figsize=(30, 6))

for img_index in range(0, 10):

    axs[0][img_index].imshow(test_images[img_index])
    axs[0][img_index].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    heatmap = latent_saliency(vae.encoder, test_images[img_index], smoothing_sigma=2.0)

    axs[1][img_index].imshow(test_images[img_index])
    axs[1][img_index].imshow(heatmap, cmap="jet", alpha=0.5)
    axs[1][img_index].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

plt.savefig("Variational Eagle/Plots/heatmap_all_smooth", bbox_inches="tight")
plt.show()

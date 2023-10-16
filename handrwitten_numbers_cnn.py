import mnist
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


# preparing the data
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_labels)

# Normalize the images (values between 0 and 1 rather than 0 and 255)
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# setting the parameters of the cnn
num_filters = 8
filter_size = 3
pool_size = 2

# build the model using those parameters
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# compile the model
model.compile(
  'adam',                               # gradient based optimiser
  loss='categorical_crossentropy',      # >2 classes???
  metrics=['accuracy'],                 # classification problem
)

# train the model
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=3,
  validation_data=(test_images, to_categorical(test_labels)),
)


# make predictions using that model (the first 10 numbers in the mnist database)
predictions = model.predict(test_images[:15])

# print the predicted and actual values
print(np.argmax(predictions, axis=1))
print(test_labels[:15])
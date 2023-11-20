from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np

image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/galedge_10.png")

# print(image[0])
# print()
# print()
# print(image[1])
# print()
# print()
# print(image[2])
# print()
# print()
print(image)
print()
print()
print(np.amax(image))
print(image.shape)      # (256, 256, 3)
print(image[0].shape)   # (256, 3)
print(image[1].shape)   # (256, 3)
print(image[2].shape)   # (256, 3)
print(image[3].shape)

fig, axs = plt.subplots(1, 4, figsize=(20,4))

axs[0].imshow(image)
axs[0].get_xaxis().set_visible(False)
axs[0].get_yaxis().set_visible(False)

axs[1].imshow(image[0])
axs[1].get_xaxis().set_visible(False)
axs[1].get_yaxis().set_visible(False)

axs[2].imshow(image[1])
axs[2].get_xaxis().set_visible(False)
axs[2].get_yaxis().set_visible(False)

axs[3].imshow(image[2])
axs[3].get_xaxis().set_visible(False)
axs[3].get_yaxis().set_visible(False)

plt.imshow(image)

plt.savefig("Plots/galedge_10")
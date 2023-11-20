from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np

image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/galedge_10.png")

print(image)
print(np.amax(image))

fig, axs = plt.subplots(1, 4, figsize=(20,4))

axs[0,0].imshow(image)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)

axs[0,1].imshow(image[0])
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)

axs[0,2].imshow(image[1])
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)

axs[0,3].imshow(image[2])
axs[0,3].get_xaxis().set_visible(False)
axs[0,3].get_yaxis().set_visible(False)

plt.imshow(image)

plt.savefig("Plots/galedge_10")
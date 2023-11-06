from matplotlib import pyplot as plt
from matplotlib import image as mpimg

image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/galedge_10.png")
plt.imshow(image)

plt.savefig("galedge_10")
import zipfile
from matplotlib import image as mpimg


# with zipfile.ZipFile("gz2_images_all.zip", "r") as zip_ref:
#     zip_ref.extractall("Galaxy Zoo Images")






image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/project/Galaxy Zoo Images/gz2_images_all/587739379910508635.jpg")

print(image.shape)
import zipfile

with zipfile.ZipFile("gz2_images_all.zip", "r") as zip_ref:
    zip_ref.extractall("Galaxy Zoo Images")
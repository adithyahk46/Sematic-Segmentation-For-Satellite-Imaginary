import os
import cv2
from PIL import Image 
import numpy as np 
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from matplotlib import pyplot as plt
import random
from pathlib import Path
import os

dataset_root_folder = Path(__file__).parent / "datasets" / "satellite"

dataset_name = "DubaiDataset"

print("Dataset Root Folder:", dataset_root_folder)

for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
    dir_name = path.split(os.path.sep)[-1]
    # print(dir_name)
    if dir_name == "images":  #change this for mask images  
        images = os.listdir(path)
        # print(images)
        for i, image_name in enumerate(images):
            if(image_name.endswith(".jpg")): #change to ".png" for mask images
                # print(image_name)
                a = True

image_patch_size = 256 #half of the original image size (512x512)

# image = cv2.imread(f'{dataset_root_folder}/{dataset_name}/Tile 2/images/image_part_001.jpg',1)
# print(image.shape)
# image_patches = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)
# print(len(image_patches))
# print(image_patches.shape)

image_dataset = []
image_extension = "jpg"  # change to "png" for mask images
for tile_id in range(1, 9):
    for image_id in range(1, 10):
        image_path = (dataset_root_folder / dataset_name / f"Tile {tile_id}" /
                      "images" / f"image_part_{image_id:03d}.{image_extension}")
        image = cv2.imread(str(image_path), 1)
        if image is not None:
            print(image_path , " = ", image.shape)
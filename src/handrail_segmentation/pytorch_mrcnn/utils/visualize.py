import cv2
import argparse
import os
import numpy as np
from PIL import Image


# Add the colored map to the image for visualization
def add_colored_to_image(image, colored):
    return cv2.addWeighted(cv2.resize(image, (colored.shape[1], colored.shape[0]))
                            .astype(np.uint8), 1,
                            colored.astype(np.uint8), .5,
                            0, cv2.CV_32F)

def convert_mask_to_image(mask, label):
    coloring_scheme = {1: [0, 39, 143],
                       2: [0, 200, 172],
                       3: [0, 106, 200],
                       4: [0, 173, 2]}
    colored_maps = np.array(Image.fromarray(mask).convert("RGB"))
    colored_maps[np.where(mask == label)] = coloring_scheme[label]
    return Image.fromarray(colored_maps)


def visualize(img_path, bbox, mask, label):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    colored_maps = convert_mask_to_image(mask, label)
    colored_image = add_colored_to_image(image, colored_map)

    path_save = 'output/'
    imageId = 'segmentation_' + os.path.splitext(image_path)[0][-7:] + '.png'
    if not cv2.imwrite(path_save + imageId, colored_image):
        raise Exception("Could not write image")

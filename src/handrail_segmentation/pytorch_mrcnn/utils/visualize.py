import cv2
import argparse
import os
import numpy as np

# Add the colored map to the image for visualization
def add_colored_to_image(image, colored):
    return cv2.addWeighted(cv2.resize(image, (colored.shape[1], colored.shape[0]))
                            .astype(np.uint8), 1,
                            colored.astype(np.uint8), .5,
                            0, cv2.CV_32F)

def visualize(img_path, bbox, mask, label):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)    

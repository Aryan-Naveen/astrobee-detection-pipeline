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
# global labels map for mouse callback
labels_map = None
# Callback when you click on any pixel in the labels_map image
# Prints the label and instance count of the clicked pixel
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # instance / panoptic
        # label id
        label = labels_map[y,x,0]
        # instance count from the other 2 values of the pixel
        instance_count = labels_map[y,x,1] * 256 + labels_map[y,x,2]
        print(f'label: {label} .. instance count: {instance_count}')

# Arg Parser to set the dataset path
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Segmentation Dataset Path')
args = parser.parse_args()
# dataset path
path = args.path
# paths of images folders
images_path = os.path.join(path, "images")
labels_map_path = os.path.join(path, "labels_maps")
colored_map_path = os.path.join(path, "colored_maps")
# list all images paths
images_names  = sorted(os.listdir(images_path))
labels_names  = sorted(os.listdir(labels_map_path))
colored_names = sorted(os.listdir(colored_map_path))
# add the root path to images names
images_paths      = [os.path.join(images_path, name) for name in images_names]
labels_map_paths  = [os.path.join(labels_map_path, name) for name in labels_names]
colored_map_paths = [os.path.join(colored_map_path, name) for name in  colored_names]
for image_path, labels_path, colored_path in zip(images_paths, labels_map_paths, colored_map_paths):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    colored_map = cv2.imread(colored_path, cv2.IMREAD_COLOR)
    colored_image = add_colored_to_image(image, colored_map)
    # print(cv2.imwrite("/home/anaveen/Documents/nasa_ws/astrobee-detection-pipeline/src/handrail_segmentation/worlds/data/data/segmentation/" + str(i) + ".png", colored_image))
    imageId = os.path.splitext(image_path)[0][-7:]
    cv2.imshow("segmentated map", colored_image)
    k = cv2.waitKey(0)
    while not k == 27:
        print(k)
        k = cv2.waitKey(0)
    if not cv2.imwrite(r'/home/anaveen/Documents/nasa_ws/astrobee-detection-pipeline/src/handrail_segmentation/worlds/data/overlayed/segmentation_' + imageId + '.png', colored_image):
        raise Exception("Could not write image")
    cv2.destroyAllWindows()

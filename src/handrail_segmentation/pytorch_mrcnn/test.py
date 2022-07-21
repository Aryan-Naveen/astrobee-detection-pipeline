import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision import transforms
import argparse

from PIL import Image
import numpy as np

from utils.visualize import visualize
from tqdm import tqdm
import os

convert_tensor = transforms.ToTensor()

def get_trained_model(weights_path, num_classes = 5):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    # replace the pre-trained head with a new one
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)


    model.load_state_dict(torch.load(weights_path))

    return model

def filter_preds(bboxs, masks, labels, scores, thresh=0.7):
    f_bboxs = []
    f_masks = []
    f_labels = []
    for bbox, mask, label, score in zip(bboxs, masks, labels, scores):
        if score > thresh:
            f_bboxs.append(bbox.reshape(4,))
            f_masks.append(mask.shape(240, 320))
            f_labels.append(label)

    return f_bboxs, f_masks, f_labels


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-i", "--img_directory", type=str, default="data_eval/images/", help="Path to image to evaluate on")
    parser.add_argument("-w", "--weights", type=str, default="checkpoints/yolov3_ckpt_140.pth")
    parser.add_argument("-n", "--nms_thesh", type=float, default=0.7)
    args = parser.parse_args()

    model = get_trained_model(args.weights)
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#    img_paths = [os.path.join(args.img_directory, img_name) for img_name in list(sorted(os.listdir(args.img_directory)))]
    img_paths = ['data_eval/images/image_0000002.png']

    for img_path in tqdm(img_paths):
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        img = [convert_tensor(img)]
        torch.cuda.synchronize()

        output = model(img)[0]
	num_detections = len(output['labels'])

        bboxs = output['boxes'].detach().numpy()
        masks = output['masks'].detach().numpy()
        labels = output['labels'].detach().numpy()
        scores = output['scores'].detach().numpy()

        bboxs, masks, labels = filter_preds(bboxs, masks, labels, scores)
        np.place(mask, mask > args.nms_thesh, output['labels'][0])
        np.place(mask, mask <= args.nms_thesh, 0)

        visualize(img_path, bbox, mask, label)


if __name__=='__main__':
    evaluate()

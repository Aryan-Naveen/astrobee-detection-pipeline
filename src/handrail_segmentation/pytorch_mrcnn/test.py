import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision import transforms
import argparse

from PIL import Image
import numpy as np

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


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-i", "--img_path", type=str, default="data/images/image_0000599.png", help="Path to image to evaluate on")
    parser.add_argument("-w", "--weights", type=str, default="checkpoints/yolov3_ckpt_140.pth")
    parser.add_argument("-n", "--nms_thesh", type=float, default=0.7)
    args = parser.parse_args()

    model = get_trained_model(args.weights)
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img = Image.open(args.img_path).convert("RGB")
    img = [convert_tensor(img)]
    torch.cuda.synchronize()

    output = model(img)[0]
    bbox = output['boxes']
    mask = output['masks'].detach().numpy().reshape(240, 320)
    np.place(mask, mask > args.nms_thesh, output['labels'][0])
    np.place(mask, mask <= args.nms_thesh, 0)
    print(np.unique(mask))


if __name__=='__main__':
    evaluate()

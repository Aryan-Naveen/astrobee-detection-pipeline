import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision import transforms
import argparse

from PIL import Image

def get_trained_model(weights_path):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    model.load_state_dict(torch.load(weights_path))

    return model


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-i", "--img_path", type=str, default="data/images/image_0000599.png", help="Path to image to evaluate on")
    parser.add_argument("-w", "--weights", type=str, default="checkpoints/yolov3_ckpt_140.pth")
    args = parser.parse_args()

    model = get_trained_model(args.weights)
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img = transforms.ToTensor(Image.open(args.img_path)).to(device)
    torch.cuda.synchronize()
    print(model(img))


if __name__=='__main__':
    evaluate()

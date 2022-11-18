import argparse
import os

import albumentations as A
import numpy as np
import torch
from torchvision.models import resnet50

from PIL import Image


class PlanigoProductClassificationInference:

    def __init__(self, device=None):

        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def inference(self, model, data):

        model = model.to(self.device).float()
        data = data.to(self.device).float()
        model.eval()  # this turns off the dropout layer and batch norm

        return model(data)


# def transform():
#     valid_transforms = A.Compose([
#         A.Rotate(always_apply=False, limit=(-10, 10), border_mode=1, p=0.2),
#         A.CLAHE(always_apply=False, p=0.8, clip_limit=(1, 4), tile_grid_size=(8, 8)),
#         A.Downscale(always_apply=True, scale_min=0.5, scale_max=0.25, interpolation=4),
#         A.GaussNoise(always_apply=False, p=0.8, var_limit=(146.37998962402344, 353.6199951171875)),
#         A.ImageCompression(always_apply=False, p=0.8, quality_lower=8, quality_upper=50, compression_type=1),
#         A.RandomBrightnessContrast(always_apply=True, brightness_limit=(-0.8, 0.8), contrast_limit=(-0.8, 0.8),
#                                    brightness_by_max=True),
#         A.OpticalDistortion(always_apply=False, p=0.8, distort_limit=(-0.5600000023841858, 0.5199999809265137),
#                             shift_limit=(-0.05000000074505806, 0.05000000074505806), interpolation=1, border_mode=0,
#                             value=(0, 0, 0), mask_value=None),
#         A.MotionBlur(always_apply=False, p=0.8, blur_limit=(5, 10)),
#         A.Resize(64, 64, always_apply=False, p=0.8),
#         A.Resize(100, 100, always_apply=False, p=0.8),
#         A.Resize(150, 150, always_apply=False, p=0.8),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)
#     ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument("--num-classes", type=int, default=1000, metavar="S", help="random seed (default: 10)")

    args = parser.parse_args()

    model = resnet50()

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
        torch.nn.Linear(in_features=1024, out_features=args.num_classes, bias=True)
    )

    # model = torch.nn.DataParallel(model)

    checkpoint = torch.load(os.path.join(args.model_dir, "last.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # test_data = get_dataloader(args.test, 32, model_dir=args.model_dir)
    test_data = []
    for image_path in os.scandir(args.test):
        test_data.append(np.rollaxis(np.array(Image.open(image_path.path).resize((224, 224))), 2, 0))

    inference = PlanigoProductClassificationInference()
    for _ in range(3):
        output = inference.inference(model, torch.from_numpy(np.array(test_data)))
        print(output.argmax(dim=1))


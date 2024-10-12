# hack to import from dir
import os, sys

sys.path.insert(0, os.getcwd())

import argparse
from options.test_options import TestOptions
from models.cut_model import CUTModel
from util.util import tensor2im

from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms as transforms


extra_args = sys.argv[5:]
sys.argv[1:] = sys.argv[1:5]



opt_cls = TestOptions()
opt = opt_cls.parse()

additional_parser = argparse.ArgumentParser(description="Additional options")
mode_group = additional_parser.add_mutually_exclusive_group(required=True)

# Single image mode
mode_group.add_argument(
    "--single_image", action="store_true", help="Use single image mode"
)
additional_parser.add_argument(
    "--image_path", type=str, help="Path to the single image file"
)

# Multiple image mode
mode_group.add_argument(
    "--multiple_images", action="store_true", help="Use multiple image mode"
)
additional_parser.add_argument(
    "--image_dir", type=str, help="Path to the directory containing multiple images"
)

# Rest of Params
additional_parser.add_argument(
    "--results_path",
    required=True,
    type=str,
    help="path to results of transformed images",
)

additional_opt, _ = additional_parser.parse_known_args(extra_args)


RESULTS_PATH = additional_opt.results_path


img2tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ]
)


def save_output(tensor: Tensor, filename: str):
    img = tensor2im(tensor)
    img = Image.fromarray(img)
    img.save(f"{RESULTS_PATH}/{filename}")


def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize([256, 256], Image.LANCZOS)
    img_tensor = img2tensor(image)
    output = transformer(img_tensor)
    filename = image_path.split("/")[-1]
    save_output(output, filename)


model = CUTModel(opt)
model.load_networks("latest")

transformer = getattr(model, "netG").to("cpu")


if additional_opt.single_image:
    transform_image(additional_opt.image_path)
else:
    IMAGES_PATH = additional_opt.image_dir

    for filename in os.listdir(IMAGES_PATH):
        transform_image(f"{IMAGES_PATH}/{filename}")

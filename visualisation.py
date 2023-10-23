import torch
import argparse
import json
import os
from timm.models import create_model
from pathlib import Path
from tqdm import tqdm
import utils
from timm.utils import accuracy
import numpy as np
import models
import models_v2
from matplotlib import pyplot as plt
import cv2
from mask_const import sample_masks, get_division_masks_for_model, DIVISION_IDS, DIVISION_MASKS
from functools import partial
from fvcore.nn import FlopCountAnalysis
import torchvision.transforms as TT
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('Script containing tasks with inference only', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    # Model parameters
    parser.add_argument('--model', default='deit_small_patch16_LS', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=1., type=float, help="Crop ratio for evaluation")

    parser.add_argument('--data-path', default=f'~/datasets/imagenet/ILSVRC/Data/CLS-LOC/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--data-split', default='val', choices=['train', 'val'],
                        type=str)
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--checkpoint', default=None, help="path to model checkpoint")
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--evaluate', nargs='*', default=[])
    parser.add_argument('--extract', nargs='*', default=[])
    parser.add_argument('--count_flops', action="store_true")

    parser.add_argument('--group_by_class', action="store_true")
    parser.add_argument('--random_masks', action="store_true")

    return parser


@torch.no_grad()
def extract(model, device, KMs, random_masks, seq: bool=False):
    # switch to evaluation mode
    model.eval()

    images = []
    for i in range(1, 5):
        image = cv2.imread(f"./data/crane/crane{i}.jpg");
        image = cv2.resize(image, (448, 448))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255
        images.append(image)
        plt.subplot(220 + i)
        plt.imshow(image)

    # We need to reorder the images to [batch, channel, widht, height]
    # The array of loaded images is [batch, height, width, channel]
    images_arr = np.stack(images)
    input_tensor = torch.Tensor(np.transpose(images_arr, [0, 3, 2, 1]))

    transform = TT.Compose([TT.Normalize(mean=0.5, std=0.2)])

    input_tensor = transform(input_tensor)


def main_setup(args):
    if args.checkpoint is None:
        print("[WARNING] --checkpoint is None")

    output_dir = Path(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=args.output_dir == "debug")

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1e3,
        img_size=args.input_size
    )
    model = model.to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        utils.interpolate_pos_embed(model, checkpoint['model'])
        msg = model.load_state_dict(checkpoint['model'])
        print("Loaded checkpoint: ", msg)

    return model


def main(args):
    model = main_setup(args)
    model = model.to(args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


import torch
import argparse
import json
import os

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from pathlib import Path
from PIL import Image
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
    parser.add_argument('--input_size', default=448, type=int, help='images input size')
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

    parser.add_argument('--output_dir', type=str, required=True,
                        help='path where to save')

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
    division_masks = get_division_masks_for_model(model)

    ret = {
        f"{k}_{m}": {"features": [], "targets": []}
        for k, m in KMs
    }

    images = []
    transform = TT.Compose([TT.ToTensor(), TT.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    for i in range(1, 5):
        image = Image.open(f"./experiments/data/crane{i}.jpg")
        image = image.resize((448, 448))
        image = image.convert("RGB")

        images.append(transform(image))

    input_tensor = torch.stack(images)

    # We need to reorder the images to [batch, channel, width, height]
    # The array of loaded images is [batch, height, width, channel]

    with torch.cuda.amp.autocast():
        for k, m in KMs:
            if random_masks:
                masks = sample_masks(division_masks, m)
            else:
                masks = division_masks[m][0]
            input_tensor = input_tensor.to(device, non_blocking=True)
            features = model.comp_forward_afterK_patches(input_tensor, K=k, masks=masks).cpu().numpy()
            ret[f"{k}_{m}"]["features"].append(features)

    return ret

def PCA_path_tokens(features):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import minmax_scale

    embed_dim = 384

    for kM in features.keys():
        patch_tokens = features[kM]['features'][0].reshape([4, embed_dim, -1])

        print('!!!')
        print(patch_tokens.shape)

        fg_pca = PCA(n_components=1)

        masks = []
        fig = plt.figure(figsize=(10, 10))

        all_patches = patch_tokens.reshape([-1, embed_dim])
        reduced_patches = fg_pca.fit_transform(all_patches)
        # scale the feature to (0,1)
        norm_patches = minmax_scale(reduced_patches)

        # reshape the feature value to the original image size
        image_norm_patches = norm_patches.reshape([4, embed_dim])

        for i in range(4):
            image_patches = image_norm_patches[i, :]

            # choose a threshold to segment the foreground
            mask = (image_patches > 0.6).ravel()
            masks.append(mask)

            image_patches[np.logical_not(mask)] = 0

            plt.subplot(221 + i)
            plt.imshow(image_patches.reshape([32, -1]).T, extent=(0, 448, 448, 0), alpha=0.5)
            fig.savefig(f"output_{kM}_{i}.png")


def extract_k16(model, device, random_masks, *args, **kwargs):
    KMs = [[k, 16] for k in range(len(model.blocks) + 1)]
    return extract(model, device, KMs=KMs, random_masks=random_masks)


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
        num_classes=1000,
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

    ret_dict = extract_k16(model, args.device, random_masks=False)

    PCA_path_tokens(ret_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


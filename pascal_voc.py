import json
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import xml.etree.ElementTree as ET

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
import models_v2
from pathlib import Path

from timm.models import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import utils


def parse_annotation_and_mask(annotation_path, masks_dir):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image_path = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    masks = []
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'name': obj_name,
            'bbox': (xmin, ymin, xmax, ymax)
        })

        mask_file = os.path.join(masks_dir, image_path.replace('.jpg', '.png'))
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)

    return {
        'image_path': image_path,
        'width': width,
        'height': height,
        'objects': objects,
        'masks': masks
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--model', default='deit_small_patch16_LS', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--checkpoint', default=None, help="path to model checkpoint")
    parser.add_argument("--image_size", default=(448, 448), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        img_size=args.image_size
    )
    model = model.to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        utils.interpolate_pos_embed(model, checkpoint['model'])
        msg = model.load_state_dict(checkpoint['model'])
        print("Loaded checkpoint: ", msg)
    # open image

    dataset_dir = '/data/pwojcik/VOCdevkit/VOC2012/'
    validation_image_set_file = os.path.join(dataset_dir, 'ImageSets/Segmentation/val.txt')
    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    masks_dir = os.path.join(dataset_dir, 'SegmentationObject')

    with open(validation_image_set_file, 'r') as f:
        validation_image_list = f.read().strip().split('\n')

    for image_id in validation_image_list:
        annotation_file = os.path.join(annotations_dir, image_id + '.xml')
        annotation_info = parse_annotation_and_mask(annotation_file, masks_dir)

        image_path = os.path.join(dataset_dir, 'JPEGImages', annotation_info['image_path'])
        print(f'Image Path: {image_path}')
        print(f'Width: {annotation_info["width"]}, Height: {annotation_info["height"]}')

        for i, obj in enumerate(annotation_info['objects']):
            print(f'Object: {obj["name"]}')
            print(f'Bounding Box: {obj["bbox"]}')
            print()
            mask = annotation_info['masks'][i]
            print(mask)
            print(f'Mask Shape: {mask.shape}')
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

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

def main():
    dataset_dir = '/data/pwojcik/VOCdevkit/VOC2012/'
    validation_image_set_file = os.path.join(dataset_dir, 'ImageSets/Segmentation/test.txt')
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
            print(f'Mask Shape: {mask.shape}')


if __name__ == '__main__':
    main()
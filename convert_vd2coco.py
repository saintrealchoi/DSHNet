import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
import csv

img_id = 0
ann_id = 0
category_dict = {}
category_instancesonly = {
    'pedestrian': 1,
    'people': 2,
    'bicycle': 3,
    'car': 4,
    'van': 5,
    'truck': 6,
    'tricycle': 7,
    'awning-tricycle': 8,
    'bus': 9,
    'motor': 10,
}
category_id = {
    0: 'ig',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others',
}
ann_dict = {}
images = []
annotations = []

data_root = 'data/VisDrone2019-DET-test-dev'
for root, dirs, files in os.walk(os.path.join(data_root, 'annotations')):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            if len(images) % 50 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))
            with open(file_path, "r") as f:
                img = Image.open(
                    os.path.join(data_root, 'images', file.split('.')[0] + '.jpg'))
                width, height = img.size[0], img.size[1]
                image = {}
                image['id'] = img_id
                img_id += 1
                image['width'] = width
                image['height'] = height
                image['file_name'] = file.split('.')[0] + '.jpg'
                images.append(image)
                for line in f.readlines():
                    line = line.strip('\n')
                    x, y, w, h, score, cls = float(line.split(',')[0]), float(line.split(',')[1]), float(
                        line.split(',')[2]), float(line.split(',')[3]), float(line.split(',')[4]), int(
                        line.split(',')[5])
                    if int(cls) not in category_id.keys():
                        continue
                    cls = category_id[int(cls)]
                    if cls not in category_instancesonly:
                        continue  # skip non-instance categories
                    xywh_box = (x, y, w, h)
                    ann = {}
                    ann['id'] = ann_id
                    ann_id += 1
                    ann['image_id'] = image['id']
                    ann['area'] = w * h
                    ann['iscrowd'] = 0
                    ann['bbox'] = xywh_box
                    ann['category_id'] = category_instancesonly[cls]
                    annotations.append(ann)
ann_dict['images'] = images
categories = [{"id": category_instancesonly[name], "name": name} for name in category_instancesonly]
ann_dict['categories'] = categories
ann_dict['annotations'] = annotations
print("Num categories: %s" % len(categories))
print("Num images: %s" % len(images))
print("Num annotations: %s" % len(annotations))
with open(os.path.join(data_root, 'coco_annotations', 'train.json'), 'w') as outfile:
    outfile.write(json.dumps(ann_dict))
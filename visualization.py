import torch
from torch import nn
from torchvision.datasets import CocoDetection
import cv2
import matplotlib.pyplot as plt
import numpy as np

testset = CocoDetection("/media/sinclair/datasets/COCO/val2017",
                         "/media/sinclair/datasets/COCO/annotations/instances_val2017.json")

def vis_coco_instance(image, label):
    """
    :param image: pil image
    :param label: COCO's stupid format, a list of dictionaries containing
    segmentation, class, bbox for every object.
    """
    image, label = testset.__getitem__(534)
    image = np.array(image)
    for i in label:
        [x,y,w,h] = i['bbox']
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255,0,0), 5)
    print("hello")
    imgplot = plt.imshow(image)
    plt.show()

def build_yolo_input_output(image, label, grid_width, anchor_box_ratios, class_mapping):
    """
    Takes in a COCO instance, and returns two tensors. One is the specified input,
    the second is the target output. They're based on the anchor boxes as well.
    :param class_mapping: a list where class_mapping[coco class] = our_class
    """


def iou(box1, box2):
    """
    in xywh format, where (x, y) is the top left corner of the box
    """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    area1 = 

vis_regularbbox()

def wato_coco_class_mapping():
    mapping = [0 for x in range(80)]



import torch
from torch import nn
from torchvision.datasets import CocoDetection

# datasets

trainset = CocoDetection("/media/sinclair/datasets/COCO/train2017",
                         "/media/sinclair/datasets/COCO/annotations/instances_train2017.json")

testset = CocoDetection("/media/sinclair/datasets/COCO/val2017",
                         "/media/sinclair/datasets/COCO/annotations/instances_val2017.json")

# dataloaders

image = trainset.__getitem__(20)[0]

image.show()
width, height = image.size

label = trainset.__getitem__(20)[1]
# in pixel image coordinates, xywh
print(width, height)
for obj in label:
    # print(obj)
    print(obj['bbox'])
    print(obj['category_id'])
# model

# optimizer

# training loop
for (i, batch) in trainloader:
    x, y = batch  # images, formatted, and target predictions
    y = classification

    x.to(device)
    optim.zero_grad()

    class_logits, bbox_activations= model(x)

    yhat = activations_to_bbox(bbox_activations)

    loc_loss = boundingboxloss()
    class_loss = nn.CrossEntropyWithLogits(class_logits, y)



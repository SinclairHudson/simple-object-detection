import torch

import torch.nn as nn

class Residual(nn.Module):
    """
    Residual block found in Darknet-53
    """
    def __init__(self, in_channels, inter_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        self.norm1 = nn.LayerNorm(inter_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        inter = self.conv1(x)
        inter = self.act(self.norm1(inter))
        inter = self.conv2(inter)
        inter = self.act(self.norm2(inter))
        return inter + x

class ResidualChain(nn.Module):

    def __init__(self, in_channels, inter_channels, num_blocks):
        super(ResidualChain, self).__init__()
        self.residualBlocks = nn.ModuleList(
            [Residual(in_channels, inter_channels) for i in range(num_blocks)]
        )
    def forward(self, x):
        for block in self.residualBlocks:
            x = block(x)
        return x


class CustomBackbone(nn.Module):
    """
    Object detection backbone, inspired by Darknet-53 from yolov3
    https://arxiv.org/pdf/1804.02767.pdf
    """
    def __init__(self, input_size, num_classes, anchor_sizes):
        """
        param anchor_sizes: a list of tuples specifying the h x w of the default bounding boxes
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.res1 = ResidualChain(64, 32, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res2 = ResidualChain(128, 64, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.res3 = ResidualChain(256, 128, 8)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.res4 = ResidualChain(512, 256, 8)
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        # self.res5 = ResidualChain(1024, 512, 4)


        self.anchors = CustomBackbone.generate_anchors(input_size, anchor_sizes)

        self.classifier = nn.Conv2d(1024 + 512, len(anchor_sizes) * (num_classes+1), kernel_size=1)

        self.localizer = nn.Conv2d(1024 + 512, len(anchor_sizes) * (4), kernel_size=1)


    @staticmethod
    def generate_anchors(input_size, anchor_sizes):
        """
        generates the anchor boxes (default boxes, priors) for object detection
        :param anchor_sizes: in pixels, the dimensions of the anchor boxes
        """
        downscale_factor = 8
        boxes_per_center = len(anchor_sizes)

        grid_width = input_size // downscale_factor

        single_grid_block_width = downscale_factor

        anchors = tf

    @staticmethod
    def activations_to_pixel_bboxes(class_activations, localization_activations, anchors):
        """
        Converts activations to bounding boxes. Importantly, this must use only
        pytorch functions, because gradients need to flow through here.
        However, not all of this needs to be exported into torchscript
        :param class_activations: B x num_anchors * (1 + num_classes) x H x W
        :param localization_activations: B x num_anchors * 4 x H x W
        :param anchors: num_anchors * 2  for just width and height

        :returns: B x num_boxes x (1 + num_classes + 4)
        """

        sigmoid = nn.Sigmoid()

        num_anchors = anchors.size[0]
        batch_size  = class_activations.size[0]
        downscale_factor = 8
        grid_width = input_size // downscale_factor

        objectness_logits = class_activations[:, :num_anchors, :, :]
        objectness_logits = objectness_logits.reshape([:, -1])
        objectness = sigmoid(objectness_logits)

        # xyhw
        x_logits = localization_activations[:, :num_anchors, :, :]
        y_logits = localization_activations[:, num_anchors:2*num_anchors, :, :]
        h_logits = localization_activations[:, 2*num_anchors:3*num_anchors, :, :]
        w_logits = localization_activations[:, 3*num_anchors:, :, :]

        x_offsets_within_gridsquare =  sigmoid(x_logits)
        y_offsets_within_gridsquare =  sigmoid(y_logits)

        x_coords = torch.arange(grid_width)
        y_coords = torch.arange(grid_width)



        grid_width = input_size // downscale_factor

        anchors = tf




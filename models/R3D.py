import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

import utils


class R3D(nn.Module):
    def __init__(self, config, num_classes):
        super(R3D, self).__init__()

        self.res3d = R3DNet(config.get('layer_sizes', (2, 2, 4, 8)))
        self.linear = nn.Linear(512, num_classes)

        utils.init_model_params(self)

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)

        return logits


class R3DNet(nn.Module):
    def __init__(self, layer_sizes):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], st_conv_block=SpatioTemporalConv)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], st_conv_block=SpatioTemporalConv, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], st_conv_block=SpatioTemporalConv, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], st_conv_block=SpatioTemporalConv, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False, st_conv_block=SpatioTemporalConv):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = st_conv_block(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = st_conv_block(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # standard conv->batchnorm->ReLU
        self.conv2 = st_conv_block(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_size, st_conv_block=SpatioTemporalConv,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = SpatioTemporalResBlock(in_channels, out_channels, kernel_size, downsample, st_conv_block=st_conv_block)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [SpatioTemporalResBlock(out_channels, out_channels, kernel_size, st_conv_block=st_conv_block)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


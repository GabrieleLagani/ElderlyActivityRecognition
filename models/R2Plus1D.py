import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

import utils
from .R3D import SpatioTemporalResLayer


class R2Plus1D(nn.Module):
    def __init__(self, config, num_classes):
        super(R2Plus1D, self).__init__()

        self.res2plus1d = R2Plus1DNet(config.get('layer_sizes', (2, 2, 4, 8)), config.get('intermed_channels', True))
        self.linear = nn.Linear(512, num_classes)

        utils.init_model_params(self)

    def forward(self, x):
        x = self.res2plus1d(x)
        logits = self.linear(x)

        return logits


class R2Plus1DNet(nn.Module):
    def __init__(self, layer_sizes, intermed_channels=True):
        super(R2Plus1DNet, self).__init__()

        SplitSpatioTemporalConv.INTERMED_CHANNELS = intermed_channels

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SplitSpatioTemporalConv(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], st_conv_block=SplitSpatioTemporalConv)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], st_conv_block=SplitSpatioTemporalConv, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], st_conv_block=SplitSpatioTemporalConv, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], st_conv_block=SplitSpatioTemporalConv, downsample=True)

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


class SplitSpatioTemporalConv(nn.Module):
    INTERMED_CHANNELS = True

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SplitSpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  (1, kernel_size[1], kernel_size[2])
        spatial_stride =  (1, stride[1], stride[2])
        spatial_padding =  (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        intermed_channels = in_channels
        if self.INTERMED_CHANNELS:
            # compute the number of intermediary channels (M) using formula from the paper section 3.5
            full_size = kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels
            split_size = kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels
            intermed_channels = full_size // split_size

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size, stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn1 = nn.BatchNorm3d(intermed_channels)

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, stride=temporal_stride, padding=temporal_padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x


if __name__ == "__main__":
    config = {'layer_sizes': (2, 2, 2, 2)}
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = R2Plus1D(config, 101)

    outputs = net(inputs)
    print(outputs.size())
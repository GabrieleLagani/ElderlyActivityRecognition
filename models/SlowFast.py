import torch
import torch.nn as nn


class SlowFast(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        blocks = config.get('layer_sizes', (2, 2, 2, 2))
        blocks = [0, 0] + list(blocks)
        self.model = SlowFastModel(num_classes=num_classes, blocks=blocks)

    def forward(self, x):
        return self.model(x)


# Model hyper-parameters
alpha = 8
beta_inv = 8
dim_inner = [0, 0, 64, 128, 256, 512]
dim_out = [0, 64, 64, 128, 256, 512]
blocks = (0, 0, 2, 2, 2, 2)


class SlowFastModel(nn.Module):
    def __init__(self, num_classes, blocks=blocks):
        super(SlowFastModel, self).__init__()
        self.stage1 = SlowFastConv()
        self.stage1_fuse = Fuse(dim_out[1] // beta_inv)

        self.stage2 = SlowFastStage(
            dim_in=(dim_out[1] + 2 * dim_out[1] // beta_inv,
                    dim_out[1] // beta_inv),
            dim_inner=(dim_inner[2], dim_inner[2] // beta_inv),
            dim_out=(dim_out[2], dim_out[2] // beta_inv),
            temp_kernel_size=(1, 3),
            stride=1,
            num_blocks=blocks[2]
        )
        self.stage2_fuse = Fuse(dim_out[2] // beta_inv)

        self.stage3 = SlowFastStage(
            dim_in=(dim_out[2] + 2 * dim_out[2] // beta_inv,
                    dim_out[2] // beta_inv),
            dim_inner=(dim_inner[3], dim_inner[3] // beta_inv),
            dim_out=(dim_out[3], dim_out[3] // beta_inv),
            temp_kernel_size=(1, 3),
            stride=2,
            num_blocks=blocks[3]
        )
        self.stage3_fuse = Fuse(dim_out[3] // beta_inv)

        self.stage4 = SlowFastStage(
            dim_in=(dim_out[3] + 2 * dim_out[3] // beta_inv,
                    dim_out[3] // beta_inv),
            dim_inner=(dim_inner[4], dim_inner[4] // beta_inv),
            dim_out=(dim_out[4], dim_out[4] // beta_inv),
            temp_kernel_size=(3, 3),
            stride=2,
            num_blocks=blocks[4]
        )
        self.stage4_fuse = Fuse(dim_out[4] // beta_inv)

        self.stage5 = SlowFastStage(
            dim_in=(dim_out[4] + 2 * dim_out[4] // beta_inv,
                    dim_out[4] // beta_inv),
            dim_inner=(dim_inner[5], dim_inner[5] // beta_inv),
            dim_out=(dim_out[5], dim_out[5] // beta_inv),
            temp_kernel_size=(3, 3),
            stride=2,
            num_blocks=blocks[5]
        )

        self.head = SlowFastHead(num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage1_fuse(x)
        x = self.stage2(x)
        x = self.stage2_fuse(x)
        x = self.stage3(x)
        x = self.stage3_fuse(x)
        x = self.stage4(x)
        x = self.stage4_fuse(x)
        x = self.stage5(x)
        x = self.head(x)
        return x


class Fuse(nn.Module):
    def __init__(self, dim_in):
        super(Fuse, self).__init__()
        self.conv = nn.Conv3d(
            dim_in,
            dim_in * 2,
            kernel_size=(5, 1, 1),
            stride=(alpha, 1, 1),
            padding=(2, 0, 0),
            bias=False
        )
        self.bn = nn.BatchNorm3d(dim_in * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_slow, x_fast = x[0], x[1]
        fuse = self.conv(x_fast)
        fuse = self.bn(fuse)
        fuse = self.act(fuse)
        x_slow = torch.cat([x_slow, fuse], 1)
        return [x_slow, x_fast]


class SlowFastConv(nn.Module):
    def __init__(self):
        super(SlowFastConv, self).__init__()

        self.slow_conv = nn.Conv3d(
            3,
            64,
            kernel_size=(1, 7, 7),
            stride=(alpha, 2, 2),
            padding=(0, 3, 3),
            bias=False
        )
        self.slow_bn = nn.BatchNorm3d(num_features=64)
        self.slow_act = nn.ReLU(inplace=True)
        self.slow_pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

        self.fast_conv = nn.Conv3d(
            3,
            64 // beta_inv,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False
        )
        self.fast_bn = nn.BatchNorm3d(num_features=64 // beta_inv)
        self.fast_act = nn.ReLU(inplace=True)
        self.fast_pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

    def forward(self, x):
        return (self.slow_pool(self.slow_act(self.slow_bn(self.slow_conv(x)))),
                self.fast_pool(self.fast_act(self.fast_bn(self.fast_conv(x)))))


class ResBlock(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 temp_kernel_size,
                 stride):
        super(ResBlock, self).__init__()

        # Number of channels and spatial size of input should be matched with output
        # because of residual connection.
        if (dim_in != dim_out) or (stride > 1):
            self.conv0 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(1, stride, stride),
                bias=False
            )
            self.conv0_bn = nn.BatchNorm3d(dim_out)

        # Tx1x1, dim_inner.
        if temp_kernel_size > 1:
            self.conv1 = nn.Conv3d(
                dim_in,
                dim_inner,
                (temp_kernel_size, 1, 1),
                padding=(temp_kernel_size//2, 0, 0),
                bias=False
            )
            self.conv1_bn = nn.BatchNorm3d(dim_inner)
            self.conv1_act = nn.ReLU(inplace=True)

        # 1x3x3, dim_inner.
        self.conv2 = nn.Conv3d(
            dim_inner if temp_kernel_size > 1 else dim_in,
            dim_inner,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False
        )
        self.conv2_bn = nn.BatchNorm3d(dim_inner)
        self.conv2_act = nn.ReLU(inplace=True)

        # 1x3x3, dim_out. Stride is applied here.
        self.conv3 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
        )
        self.conv3_bn = nn.BatchNorm3d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        f_x = x
        if hasattr(self, 'conv1'):
            f_x = self.conv1(f_x)
            f_x = self.conv1_bn(f_x)
            f_x = self.conv1_act(f_x)
        f_x = self.conv2(f_x)
        f_x = self.conv2_bn(f_x)
        f_x = self.conv2_act(f_x)
        f_x = self.conv3(f_x)
        f_x = self.conv3_bn(f_x)

        if hasattr(self, 'conv0'):
            x = self.conv0_bn(self.conv0(x))
        x = x + f_x
        x = self.act(x)

        return x


class SlowFastStage(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 temp_kernel_size,
                 stride,
                 num_blocks):
        super(SlowFastStage, self).__init__()
        self.num_blocks = num_blocks

        for pathway in range(2):
            for i in range(num_blocks):
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_inner[pathway],
                    dim_out[pathway],
                    temp_kernel_size[pathway],
                    stride if i == 0 else 1
                )
                self.add_module('res_' + ('fast' if pathway else 'slow') + f'_{i}', res_block)

    def forward(self, x):
        for pathway in range(2):
            for i in range(self.num_blocks):
                m = getattr(self, 'res_' + ('fast' if pathway else 'slow') + f'_{i}')
                x[pathway] = m(x[pathway])

        return x


class SlowFastHead(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(SlowFastHead, self).__init__()
        self.slow_pool = nn.AdaptiveAvgPool3d(1)
        self.fast_pool = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(dim_out[-1] + dim_out[-1] // beta_inv, num_classes)

    def forward(self, x):
        x[0] = self.slow_pool(x[0])
        x[0] = x[0].view(x[0].shape[0], -1)
        x[1] = self.fast_pool(x[1])
        x[1] = x[1].view(x[1].shape[0], -1)
        x = torch.cat([x[0], x[1]], 1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x

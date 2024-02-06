import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import EncoderBlock

import utils


class TubeViT(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        clip_len = config.get('clip_len', 16)
        img_size = config.get('crop_size', 112)
        embed_dim = config.get('embed_dim', 768)
        ff_mult = config.get('ff_mult', 4)
        drop = config.get('drop', 0.)
        depth, heads = config.get('layer_sizes', (12, 12))
        norm = utils.retrieve(config.get('norm', 'torch.nn.LayerNorm'))

        self.model = TubeViTModel(num_classes=num_classes, video_shape=[clip_len, img_size, img_size],
                                  num_layers=depth, num_heads=heads, embed_dim=embed_dim, ff_mult=ff_mult,
                                  dropout=drop, attention_dropout=drop, norm_layer=norm)

        self.disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', True)

    def forward(self, x):
        return self.model(x)

    def get_train_params(self):
        return utils.disable_wd_for_pos_emb(self)


class TubeViTModel(nn.Module):
    BASE_VIDEO_SHAPE = (64, 224, 224)
    def __init__(
        self,
        num_classes,
        in_chans=3,
        video_shape=(64, 224, 224),
        num_layers=12,
        num_heads=12,
        embed_dim=768,
        ff_mult=4,
        dropout=0.0,
        attention_dropout=0.0,
        norm_layer=nn.LayerNorm,
        representation_size=None,
    ):
        super(TubeViTModel, self).__init__()
        self.video_shape = video_shape
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        kernel_sizes = (
            (8, 8, 8),
            (16, 4, 4),
            (4, 12, 12),
            (1, 16, 16),
        )
        strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
        )
        offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )
        self.kernel_sizes, self.strides, self.offsets = self._adapt_shapes(kernel_sizes), self._adapt_shapes(strides), self._adapt_shapes(offsets)
        self.sparse_tubes_tokenizer = SparseTubesTokenizer(in_chans, self.embed_dim, self.kernel_sizes, self.strides, self.offsets)

        self.pos_embed = torch.nn.Parameter(self._generate_position_embedding(), requires_grad=False)
        self.class_token = nn.Parameter(utils.trunc_normal_(torch.empty(1, 1, self.embed_dim), std=.02), requires_grad=True)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=self.embed_dim,
            ff_mult=ff_mult,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer
        )

        self.attention_pooling = SelfAttentionPooling(self.embed_dim)

        heads_layers = []
        if representation_size is None:
            heads_layers.append(nn.Linear(self.embed_dim, self.num_classes))
        else:
            heads_layers.append(nn.Linear(self.embed_dim, representation_size))
            heads_layers.append(nn.Tanh())
            heads_layers.append(nn.Linear(representation_size, self.num_classes))
        self.heads = nn.Sequential(*heads_layers)

    def forward(self, x):
        x = self.sparse_tubes_tokenizer(x)
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.attention_pooling(x)
        x = self.heads(x)

        return x

    def _adapt_shapes(self, shapes):
        shapes = list(shapes)
        for i in range(len(shapes)):
            shapes[i] = [s if j != 0 else (((s * self.video_shape[j]) + ((-s * self.video_shape[j]) % self.BASE_VIDEO_SHAPE[j])) // self.BASE_VIDEO_SHAPE[j]) for j, s in enumerate(shapes[i])]
        return shapes

    def _calc_conv_shape(self, kernel_size, stride, offset):
        return [1 + (self.video_shape[i] - offset[i] - kernel_size[i]) // stride[i] for i in range(len(self.video_shape))]

    def _generate_position_embedding(self):
        position_embedding = [torch.zeros(1, self.embed_dim).float()]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0)
        return position_embedding


class SparseTubesTokenizer(nn.Module):
    def __init__(self, in_chans, hidden_dim, kernel_sizes, strides, offsets):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.offsets = offsets

        self.conv_proj_weight = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.hidden_dim, self.in_chans, *self.kernel_sizes[0]))), requires_grad=True)
        self.conv_proj_bias = nn.Parameter(torch.zeros(len(self.kernel_sizes), self.hidden_dim), requires_grad=True)

    def forward(self, x):
        b, c, t, h, w = x.shape
        tubes = []
        for i in range(len(self.kernel_sizes)):
            if i == 0:
                weight = self.conv_proj_weight
            else:
                weight = F.interpolate(self.conv_proj_weight, self.kernel_sizes[i], mode='trilinear')

            tube = F.conv3d(
                x[:, :, self.offsets[i][0] :, self.offsets[i][1] :, self.offsets[i][2] :],
                weight,
                bias=self.conv_proj_bias[i],
                stride=self.strides[i],
            )

            tube = tube.reshape((b, self.hidden_dim, -1))

            tubes.append(tube)

        x = torch.cat(tubes, dim=-1)
        x = x.permute(0, 2, 1)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        embed_dim,
        ff_mult,
        dropout,
        attention_dropout,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers = []
        for i in range(num_layers):
            layers.append(EncoderBlock(num_heads, embed_dim, ff_mult * embed_dim, dropout, attention_dropout, norm_layer))
        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        return self.norm(self.layers(self.dropout(x)))

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        att_w = nn.functional.softmax(self.W(x), dim=1)
        x = torch.sum(x * att_w, dim=1)
        return x


def get_3d_sincos_pos_embed(embed_dim, tube_shape, stride, offset, kernel_size):
    embed_dim_spatial = 2 * embed_dim // 3
    embed_dim_temporal = embed_dim // 3

    # spatial
    grid_h_size = tube_shape[1]
    grid_h = torch.arange(grid_h_size).float()
    grid_h = grid_h * stride[1] + offset[1] + (kernel_size[1] - 1) // 2

    grid_w_size = tube_shape[2]
    grid_w = torch.arange(tube_shape[2]).float()
    grid_w = grid_w * stride[2] + offset[2] + (kernel_size[2] - 1) // 2

    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_spatial // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_spatial // 2, grid[1])
    pos_embed_spatial = torch.cat([emb_h, emb_w], dim=1)

    # temporal
    t_size = tube_shape[0]
    grid_t = torch.arange(t_size).float()
    grid_t = grid_t * stride[0] + offset[0] + (kernel_size[0] - 1) // 2
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    pos_embed_temporal = torch.repeat_interleave(pos_embed_temporal.unsqueeze(1), grid_h_size * grid_w_size, dim=1)
    pos_embed_spatial = torch.repeat_interleave(pos_embed_spatial.unsqueeze(0), t_size, dim=0)

    pos_embed = torch.cat([pos_embed_temporal, pos_embed_spatial], dim=-1)
    pos_embed = pos_embed.reshape(-1, embed_dim)

    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = torch.arange(embed_dim // 2).float()
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = pos.unsqueeze(1) * omega.unsqueeze(0)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


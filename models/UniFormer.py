import torch
import torch.nn as nn

import utils


class UniFormer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        layer_sizes = config.get('layer_sizes', (3, 4, 8, 3))
        dim_head = config.get('embed_dim', 64)
        patch_size = config.get('patch_size', 4)
        window_size = config.get('window_size', 5)
        ff_mult = config.get('ff_mult', 4)
        dropout = config.get('drop', 0.)
        norm = utils.retrieve(config.get('norm', 'models.UniFormer.LayerNorm'))

        self.model = UniFormerModel(num_classes=num_classes, depths=layer_sizes, patch_size=patch_size, local_aggr_size=window_size,
                                    dim_head=dim_head, ff_mult=ff_mult, attn_dropout=dropout, ff_dropout=dropout, norm_layer=norm)

        self.disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', True)

    def forward(self, x):
        return self.model(x)

    def get_train_params(self):
        return utils.disable_wd_for_pos_emb(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.weight + self.bias


class UniFormerModel(nn.Module):
    def __init__(
        self,
        num_classes,
        in_chans=3,
        dims=(64, 128, 256, 512),
        depths=(3, 4, 8, 3),
        mhsa_types=('l', 'l', 'g', 'g'),
        local_aggr_size=5,
        patch_size=4,
        ff_mult=4,
        dim_head=64,
        ff_dropout=0.,
        attn_dropout=0.,
        norm_layer=LayerNorm
    ):
        super().__init__()
        mhsa_types = [t.lower() for t in mhsa_types]

        self.to_tokens = nn.Conv3d(in_chans, dims[0], (3, patch_size, patch_size), stride=(2, patch_size, patch_size), padding=(1, 0, 0))

        self.stages = nn.ModuleList([])

        for i in range(len(depths)):
            is_last = True if i == (len(depths) - 1) else False
            stage_dim = dims[i]
            depth = depths[i]
            mhsa_type = mhsa_types[i]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Transformer(
                    dim=stage_dim,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    mhsa_type=mhsa_type,
                    local_aggr_size=local_aggr_size,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                    norm_layer=norm_layer
                ),
                nn.Sequential(
                    nn.Conv3d(stage_dim, dims[i + 1], (1, 2, 2), stride=(1, 2, 2)),
                    norm_layer(dims[i + 1]),
                ) if not is_last else None
            ]))

        self.norm = norm_layer(dims[-1])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.clf = nn.Linear(dims[-1], num_classes)

    def forward(self, video):
        x = self.to_tokens(video)

        for transformer, conv in self.stages:
            x = transformer(x)

            if conv is not None:
                x = conv(x)

        x = self.pool(self.norm(x)).reshape(x.shape[0], -1)

        return self.clf(x)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mhsa_type='g',
        local_aggr_size=5,
        dim_head=64,
        ff_mult=4,
        ff_dropout=0.,
        attn_dropout=0.,
        norm_layer=LayerNorm
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mhsa_type == 'l':
                attn = LocalAggr(dim, heads=heads, dim_head=dim_head, local_aggr_size=local_aggr_size)
            elif mhsa_type == 'g':
                attn = GlobalAggr(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, norm_layer=norm_layer)
            else:
                raise NotImplementedError("Attention type {} not available".format(mhsa_type))

            self.layers.append(nn.ModuleList([
                nn.Conv3d(dim, dim, 3, padding=1),
                attn,
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout, norm_layer=norm_layer),
            ]))

    def forward(self, x):
        for conv, attn, ff in self.layers:
            x = conv(x) + x
            x = attn(x) + x
            x = ff(x) + x
        return x


def FeedForward(dim, mult=4, dropout=0., norm_layer=LayerNorm):
    return nn.Sequential(
        norm_layer(dim),
        nn.Conv3d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv3d(dim * mult, dim, 1)
    )


class LocalAggr(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head=64,
        local_aggr_size=5,
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        # Uses batch norm here
        self.norm = nn.BatchNorm3d(dim)

        # only values, as the attention matrix is taken care of by a convolution
        self.to_v = nn.Conv3d(dim, inner_dim, 1, bias=False)

        # Aggregate tokens using fixed weights which depend on the relative distance from a given token, instead of a
        # less efficient attention mask. Easily implemented as a grouped convolution.
        self.rel_pos = nn.Conv3d(heads, heads, local_aggr_size, padding=(local_aggr_size - 1) // 2, groups=heads)

        # combine out across all the heads
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        b, c, h = x.shape[0], x.shape[1], self.heads

        # to values
        v = self.to_v(x)

        # split out heads
        v = v.reshape(b, h, (c // h), *v.shape[2:]).transpose(1, 2).reshape(b * (c // h), h, *v.shape[2:])

        # aggregate by relative positions
        out = self.rel_pos(v)

        # combine heads
        out = out.reshape(b, c // h, h, *out.shape[2:]).transpose(1, 2).reshape(b, c, *out.shape[2:])
        return self.to_out(out)

class GlobalAggr(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head=64,
        dropout=0.,
        norm_layer=LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = norm_layer(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)

        b, c, h = x.shape[0], x.shape[1], self.heads

        q, k, v = self.to_qkv(x.reshape(b, c, -1)).reshape(b, 3, h, c // h, -1).transpose(-2, -1).chunk(3, dim=1)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = self.to_out(out.transpose(-2, -1).reshape(b, c, -1)).reshape(b, c, *x.shape[2:])
        return out


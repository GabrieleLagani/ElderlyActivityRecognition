import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _triple

from .STAM import token_pool, Mlp
import utils


class ViViT(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        model_version = config.get('model_version', 1)

        clip_len = config.get('clip_len', 16)
        img_size = config.get('crop_size', 112)
        patch_size = config.get('patch_size', (4, 8, 8))
        embed_dim = config.get('embed_dim', 192)
        ff_mult = config.get('ff_mult', 4)
        drop = config.get('drop', 0.)
        token_pool = config.get('token_pool', 'first')
        s_depth, s_heads, t_depth, t_heads = config.get('layer_sizes', (4, 3, 4, 3))
        norm = utils.retrieve(config.get('norm', 'torch.nn.LayerNorm'))

        self.model = ViViTModel(num_classes=num_classes, image_size=img_size, num_frames=clip_len, tubelet_size=patch_size,
                                dim=embed_dim, s_depth=s_depth, s_heads=s_heads, t_depth=t_depth, t_heads=t_heads,
                                ff_mult=ff_mult, pool_type=token_pool, dropout=drop, emb_dropout=drop, norm_layer=norm,
                                vivit_model=model_version)

        self.disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', True)

    def forward(self, x):
        return self.model(x)

    def get_train_params(self):
        return utils.disable_wd_for_pos_emb(self)


class ViViTModel(nn.Module):
    def __init__(self, num_classes, in_chans=3, image_size=112, num_frames=16, tubelet_size=(4, 16, 16),
                 dim=256, s_depth=4, s_heads=4, t_depth=4, t_heads=4, norm_layer=nn.LayerNorm,
                 ff_mult=4, pool_type='first', dropout=0., emb_dropout=0., vivit_model=1):
        super().__init__()
        if vivit_model not in [1, 2, 3, 4]:
            raise NotImplementedError("ViViT model {} not available".format(vivit_model))
        self.vivit_model = vivit_model

        tubelet_size = _triple(tubelet_size)
        image_size = _pair(image_size)
        num_frames = num_frames // tubelet_size[0]
        num_patches = (image_size[0] // tubelet_size[1]) * (image_size[1] // tubelet_size[2])
        self.tubelet_embed = nn.Conv3d(in_chans, dim, tubelet_size, stride=tubelet_size)

        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, s_heads, dim * ff_mult, s_depth, dropout, norm_layer=norm_layer, vivit_model=self.vivit_model)

        if self.vivit_model == 2:
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.temporal_transformer = Transformer(dim, t_heads, dim * ff_mult, t_depth, dropout, norm_layer=norm_layer, vivit_model=self.vivit_model)

        if self.vivit_model == 3 or self.vivit_model == 4:
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.pool_type = pool_type

        self.mlp_head = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.tubelet_embed(x)
        b, c, t, h, w = x.shape
        x = x.transpose(1, 2).reshape(b, t, c, -1).transpose(-2, -1)

        x = x + self.pos_embed
        x = self.dropout(x)

        if self.vivit_model == 1:
            x = x.reshape(b, -1, c)
        else:
            x = x.reshape(b * t, -1, c)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.vivit_model == 3 or self.vivit_model == 4:
            x = x.reshape(b, t, -1, c)
            x = torch.cat((self.temporal_cls_token.expand(x.shape[0], -1, x.shape[2], -1), x), dim=1)

        x = self.transformer(x)

        if self.vivit_model == 2:
            x = token_pool(x, pool_type=self.pool_type).reshape(b, t, c)
            x = torch.cat((self.temporal_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.temporal_transformer(x)

        if self.vivit_model == 3 or self.vivit_model == 4:
            x = token_pool(x, pool_type=self.pool_type)

        x = token_pool(x, pool_type=self.pool_type)

        return self.mlp_head(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, depth, dropout=0., norm_layer=nn.LayerNorm, vivit_model=1):
        super().__init__()
        self.vivit_model = vivit_model
        self.layers = nn.ModuleList([])
        self.norm = norm_layer(dim)
        attn_block = FDotAttention if self.vivit_model == 4 else Attention
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn_block(dim, heads=heads, dropout=dropout), norm_layer=norm_layer),
                PreNorm(dim, attn_block(dim, heads=heads, dropout=dropout), norm_layer=norm_layer) if self.vivit_model == 3 else None,
                PreNorm(dim, Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=nn.GELU, drop=dropout), norm_layer=norm_layer)
            ]))

    def forward(self, x):
        for attn, t_attn, ff in self.layers:
            if self.vivit_model == 3:
                b, t, n, c = x.shape
                x = x.reshape(b * t, n, c)

                # Spatial attention
                x = attn(x) + x
                x = x.reshape(b, t, n, c).transpose(1, 2).reshape(b * n, t, c)

                # Temporal attention
                x = t_attn(x) + x
                x = x.reshape(b, n, t, c).transpose(1, 2)
            else:
                x = attn(x) + x

            x = ff(x) + x
        return self.norm(x)


def qkv_aggregate(q, k, v):
    scale = v.shape[-1] ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    return (attn @ v)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, c, h = x.shape[0], x.shape[1], x.shape[2], self.heads

        qkv = self.to_qkv(x).reshape(b, -1, 3, h, c // h).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        out = qkv_aggregate(q, k, v)

        out = self.to_out(out.transpose(1, 2).reshape(b, -1, c))
        return out

class FDotAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, t, n, c, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.heads

        qkv = self.to_qkv(x).reshape(b, -1, 3, h, c // h).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        qs, qt = q.chunk(2, dim=1)
        ks, kt = k.chunk(2, dim=1)
        vs, vt = v.chunk(2, dim=1)

        # Spatial attention
        qs, ks, vs = (s.reshape(b, h // 2, t, n, c // h).permute(0, 2, 1, 3, 4).reshape(b * t, h // 2, n, c // h) for s in (qs, ks, vs))
        s_out = qkv_aggregate(qs, ks, vs)
        s_out = s_out.reshape(b, t, h // 2, n, c // h).permute(0, 2, 1, 3, 4)

        # Temporal attention
        qt, kt, vt = (s.reshape(b, h // 2, t, n, c // h).permute(0, 3, 1, 2, 4).reshape(b * n, h // 2, t, c // h) for s in (qt, kt, vt))
        t_out = qkv_aggregate(qt, kt, vt)
        t_out = t_out.reshape(b, n, h // 2, t, c // h).permute(0, 2, 3, 1, 4)

        out = torch.cat((s_out, t_out), dim=1)
        out = self.to_out(out.permute(0, 2, 3, 1, 4).reshape(b, t, n, c))
        return out



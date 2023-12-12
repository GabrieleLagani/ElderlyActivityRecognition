import torch
import torch.nn as nn

import utils


class STAM(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        clip_len = config.get('clip_len', 16)
        img_size = config.get('crop_size', 112)
        patch_size = config.get('patch_size', (1, 8, 8))
        embed_dim = config.get('embed_dim', 768)
        ff_mult = config.get('ff_mult', 4)
        drop = config.get('drop', 0.)
        drop_path = config.get('drop_path', 0.)
        token_pool = config.get('token_pool', 'first')
        depth, heads, a_depth, a_heads = config.get('layer_sizes', (12, 12, 6, 8))
        aggr_hidden_dim = config.get('aggr_hidden_dim', 2048)
        norm = utils.retrieve(config.get('norm', 'torch.nn.LayerNorm'))

        self.model = VisionTransformer(img_size=img_size, clip_len=clip_len, num_classes=num_classes, token_pool=token_pool,
            patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=heads, ff_mult=ff_mult,
            aggr=(a_depth, a_heads), aggr_hidden_dim=aggr_hidden_dim, drop_path_rate=drop_path, drop_rate=drop, attn_drop_rate=drop, norm_layer=norm)

        self.disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', True)

    def forward(self, x):
        return self.model(x)

    def get_train_params(self):
        return utils.disable_wd_for_pos_emb(self)


class VisionTransformer(nn.Module):
    ATTN_TYPE_SPACE = 'space'
    ATTN_TYPE_TIME = 'time'
    ATTN_TYPE_JOINT = 'joint'
    ATTN_TYPE_DIVIDED = 'divided'

    def __init__(self, img_size=224, clip_len=16, patch_size=(1, 16, 16), in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, ff_mult=4, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, token_pool='first',
                 aggr=None, aggr_hidden_dim=2048, attn_type=ATTN_TYPE_SPACE, reset_temporal_weights=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(clip_len=clip_len, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        num_temporal_patches = self.patch_embed.num_temporal_patches

        self.attn_type = attn_type
        if self.attn_type not in [self.ATTN_TYPE_SPACE, self.ATTN_TYPE_TIME, self.ATTN_TYPE_JOINT, self.ATTN_TYPE_DIVIDED]:
            raise NotImplementedError("Attention type {} not available".format(self.attn_type))

        self.cls_token = nn.Parameter(utils.trunc_normal_(torch.empty(1, 1, embed_dim), std=.02), requires_grad=True)
        if self.attn_type != self.ATTN_TYPE_TIME:
            self.pos_embed = nn.Parameter(utils.trunc_normal_(torch.empty(1, num_patches, embed_dim), std=.02), requires_grad=True)
        if self.attn_type != self.ATTN_TYPE_SPACE:
            self.temporal_pos_embed = nn.Parameter(utils.trunc_normal_(torch.empty(1, num_temporal_patches, embed_dim), std=.02), requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, ff_mult=ff_mult, num_temporal_patches=num_temporal_patches, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, divided_attn=self.attn_type==self.ATTN_TYPE_DIVIDED)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Aggregation block
        self.token_pool = token_pool
        self.aggregate = None
        if aggr is not None:
            if self.attn_type not in [self.ATTN_TYPE_SPACE, self.ATTN_TYPE_TIME]:
                raise NotImplementedError("Aggregation not available for selected attention type ({})".format(self.attn_type))
            self.aggregate = Aggregate(num_temporal_patches if self.attn_type == self.ATTN_TYPE_SPACE else num_patches,
                                       embed_dim=embed_dim, depth=aggr[0], num_heads=aggr[1], hidden_dim=aggr_hidden_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        utils.init_model_params(self, mode='trunc_normal')

        if reset_temporal_weights and self.attn_type == self.ATTN_TYPE_DIVIDED:
            self.init_temporal_weights()

    def init_temporal_weights(self):
        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str:
                if i > 0:
                  nn.init.constant_(m.temporal_fc.weight, 0)
                  nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        utils.init_model_params(self.head, mode='trunc_normal')

    def forward_features(self, x):
        B, _, T, H, W = x.shape
        x = self.patch_embed(x)
        N, C = x.shape[1], x.shape[2]

        # Spatial embedding
        if self.attn_type != self.ATTN_TYPE_TIME:
            x = x + self.pos_embed
            x = self.dropout(x)

        # Temporal embedding
        if self.attn_type != self.ATTN_TYPE_SPACE:
            x = x.reshape(B, T, N, C).permute(0, 2, 1, 3).reshape(B*N, T, C)
            x = x + self.temporal_pos_embed
            x = self.dropout(x)
            if self.attn_type != self.ATTN_TYPE_TIME:
                x = x.reshape(B, N, T, C).permute(0, 2, 1, 3)
                if self.attn_type == self.ATTN_TYPE_JOINT: x = x.reshape(B, T*N, C)
                elif self.attn_type == self.ATTN_TYPE_DIVIDED: x = x.reshape(B*T, N, C)

        # Prepend cls_token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        # Token aggregation
        x = token_pool(self.norm(x), pool_type=self.token_pool)
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.aggregate is not None:
            x = self.aggregate(x)
        else:
            x = x.mean(dim=1)
        x = self.head(x)
        return x


class Aggregate(nn.Module):
    def __init__(self, size=None, embed_dim=768, depth=6, num_heads=8, hidden_dim=2048, token_pool='first'):
        super(Aggregate, self).__init__()
        self.size = size
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim))

        self.cls_token = nn.Parameter(utils.trunc_normal_(torch.empty(1, 1, embed_dim), std=.02), requires_grad=True)
        self.pos_embed = nn.Parameter(utils.trunc_normal_(torch.empty(1, self.size, embed_dim), std=.02), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.token_pool = token_pool

        utils.init_model_params(self, mode='trunc_normal')

    def forward(self, x):
        x = x + self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return token_pool(self.transformer_enc(x.transpose(1, 0)).transpose(1, 0), pool_type=self.token_pool)


class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.norm(x.transpose(-2, -1)).transpose(-2, -1)

class PatchEmbed(nn.Module):
    def __init__(self, clip_len=16, img_size=224, patch_size=(1, 16, 16), in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        num_patches = (img_size // patch_size[1]) * (img_size // patch_size[2])
        num_temporal_patches = clip_len // patch_size[0]
        self.img_size = img_size
        self.clip_len = clip_len
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_temporal_patches = num_temporal_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0] * x.shape[2], -1, x.shape[1])
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, num_temporal_patches=16, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, divided_attn=False):
        super().__init__()
        self.num_temporal_patches = num_temporal_patches

        self.divided_attn = divided_attn
        if self.divided_attn:
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=ff_mult*dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.divided_attn:
            nvids = x.shape[0] // self.num_temporal_patches
            cls_token, x = x[:, 0, :].unsqueeze(1), x[:, 1:, :]

            # Temporal attention
            t_attn = x.reshape(nvids, self.num_temporal_patches, -1, x.shape[-1]).transpose(1, 2).reshape(-1, self.num_temporal_patches, x.shape[-1])
            t_attn = self.temporal_fc(self.drop_path(self.temporal_attn(self.temporal_norm1(t_attn))))
            x = x + t_attn.reshape(nvids, -1, self.num_temporal_patches, x.shape[-1]).transpose(1, 2).reshape(nvids * self.num_temporal_patches, -1, x.shape[-1])
            x = torch.cat([cls_token, x], dim=1)

            # Spatial attention
            s_attn = self.drop_path(self.attn(self.norm1(x))).reshape(nvids, self.num_temporal_patches, -1, x.shape[-1])
            s_attn= torch.cat((s_attn[:, :, 0, :].unsqueeze(2).mean(dim=1, keepdim=True).expand(-1, self.num_temporal_patches, -1, -1), s_attn[:, :, 1:, :]), dim=2)
            return x + s_attn.reshape(nvids * self.num_temporal_patches, -1, x.shape[-1])

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(x, drop_prob=0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def token_pool(x, pool_type='first'):
    if pool_type == 'first': return x[:, 0]
    if pool_type == 'avg': return x.mean(dim=1)
    if pool_type == 'max': return x.max(dim=1)[0]
    raise NotImplementedError("Token pooling with pool type {} unsupported".format(pool_type))


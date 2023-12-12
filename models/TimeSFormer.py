import torch
import torch.nn as nn

from .STAM import VisionTransformer
import utils

class TimeSFormer(nn.Module):
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
        depth, heads = config.get('layer_sizes', (12, 12))
        attn_type = config.get('attn_type', 'divided')
        norm = utils.retrieve(config.get('norm', 'torch.nn.LayerNorm'))

        self.model = VisionTransformer(img_size=img_size, clip_len=clip_len, num_classes=num_classes, token_pool=token_pool,
            patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=heads, ff_mult=ff_mult, attn_type=attn_type,
            drop_path_rate=drop_path, drop_rate=drop, attn_drop_rate=drop, norm_layer=norm)

        self.disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', True)

    def forward(self, x):
        x = self.model(x)
        return x

    def get_train_params(self):
        return utils.disable_wd_for_pos_emb(self)

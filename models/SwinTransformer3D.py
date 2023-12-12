import torch
import torch.nn as nn
import torch.nn.functional as F

from .STAM import DropPath, Mlp
import utils


class SwinTransformer3D(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        patch_size = config.get('patch_size', (4, 4, 4))
        window_size = config.get('window_size', (2, 7, 7))
        embed_dim = config.get('embed_dim', 32)
        ff_mult = config.get('ff_mult', 4)
        drop = config.get('drop', 0.)
        drop_path = config.get('drop_path', 0.)
        layer_sizes = config.get('layer_sizes', (2, 3, 2, 6, 6, 12, 2, 24))
        depths = [layer_sizes[i] for i in range(len(layer_sizes)) if i % 2 == 0]
        heads = [layer_sizes[i] for i in range(len(layer_sizes)) if i % 2 == 1]
        norm = utils.retrieve(config.get('norm', 'torch.nn.LayerNorm'))

        self.model = SwinTransformer3DModel(num_classes=num_classes, patch_size=patch_size, embed_dim=embed_dim,
            depths=depths, num_heads=heads, window_size=window_size, ff_mult=ff_mult,
            drop_rate=drop, attn_drop_rate=drop, drop_path_rate=drop_path, norm_layer=norm)

        self.disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', True)

    def forward(self, x):
        return self.model(x)

    def get_train_params(self):
        return utils.disable_wd_for_pos_emb(self)


class SwinTransformer3DModel(nn.Module):
    def __init__(self,
                 num_classes,
                 in_chans=3,
                 patch_size=(4, 4, 4),
                 embed_dim=32,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 ff_mult=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=num_heads[0]*embed_dim, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        for i in range(self.num_layers):
            self.layers.append(SwinTransformer3DStage(
                dim=num_heads[i if i < len(num_heads) else -1]*embed_dim,
                out_dim=num_heads[i+1 if i+1 < len(num_heads) else -1]*embed_dim,
                depth=depths[i],
                num_heads=num_heads[i if i < len(num_heads) else -1],
                window_size=window_size,
                mlp_ratio=ff_mult,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i < self.num_layers - 1 else None))

        self.features = num_heads[-1] * embed_dim
        self.num_classes = num_classes

        self.norm = norm_layer(self.features)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.clf = nn.Linear(self.features, self.num_classes)

    def forward(self, x):
        # Prepare input
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # Extract features from Swin Transformer stages
        for l in self.layers:
            x = l(x)

        # Classifier head
        x = self.norm(x).permute(0, 4, 1, 2, 3)
        x = self.pool(x).reshape(x.shape[0], -1)
        x = self.clf(x)

        return x

# Map patch extracted from an input tensor to a token representation.
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(self.in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, T, H, W = x.shape

        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x).permute(0, 2, 3, 4, 1)

        if self.norm is not None:
            x = self.norm(x)

        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.reduction = nn.Linear(4 * self.dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * self.dim)

    def forward(self, x):
        B, T, H, W, C = x.shape

        if (H % 2 != 0) or (W % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)

        return x

# Applies attention in local windows extracted from the input tensor.
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_shape, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_shape = window_shape
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.rel_pos = RelativePosEmb(self.num_heads, self.window_shape)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bound_window_shape=(2, 7, 7), mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply relative positional embedding
        attn = self.rel_pos(attn, bound_window_shape)

        # Apply mask:
        # Each Swim Transformer layer applies attention on local windows extracted from an input tensor. In the next layer, the
        # input tensor is shifted by half a window on each direction, and the local attention mechanism is applied again, so
        # that information can also be aggregated across neighboring patches. However, when the tensor is shifted with the
        # torch.roll function, the leftmost columns are reinserted at the rightmost position. In order to prevent non-local
        # attentional aggregation at those position, the attention mask assigns them a low attention score.
        if mask is not None:
            L = mask.shape[0] # Number of window locations extracted from the input
            attn = attn.reshape(B // L, L, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        # Attention aggregate
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# Relative position embedding computes the relative position between any two locations in a given input window, and
# assigns a relative position embedding to each pair of tokens based on their relative position.
class RelativePosEmb(nn.Module):
    def __init__(self, num_heads, window_shape):
        super().__init__()

        self.num_heads = num_heads
        self.window_shape = window_shape
        M = (2 * window_shape[0] - 1) * (2 * window_shape[1] - 1) * (2 * window_shape[2] - 1)

        # Assign 3D coordinates to each window location
        coords_t = torch.arange(self.window_shape[0])
        coords_h = torch.arange(self.window_shape[1])
        coords_w = torch.arange(self.window_shape[2])
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij'))

        # Compute relative positions for each pair of window locations
        rel_pos = coords.reshape(3, -1, 1) - coords.reshape(3, 1, -1)

        # Assign each relative position to an index between 0 and M
        rel_pos_idx = self.rel_pos2idx(rel_pos)
        self.register_buffer("rel_pos_idx", rel_pos_idx.reshape(*self.window_shape, *self.window_shape))

        # relative positional embeddings
        rel_pos_emb = utils.trunc_normal_(torch.empty(num_heads, M), std=.02)
        self.rel_pos_embed = nn.Parameter(rel_pos_emb)

    def rel_pos2idx(self, rel_pos):
        # shift relative position coordinates to start from 0
        for i in range(len(self.window_shape)): rel_pos[i] += self.window_shape[i] - 1

        # reweight relative coordinate offsets
        rel_pos[0] *= (2 * self.window_shape[1] - 1) * (2 * self.window_shape[2] - 1)
        rel_pos[1] *= (2 * self.window_shape[2] - 1)

        return rel_pos.sum(0)

    def forward(self, x, bound_window_shape=(2, 7, 7)):
        T, H, W = bound_window_shape
        rel_pos_idx = self.rel_pos_idx[:T, :H, :W, :T, :H, :W].reshape(-1)
        return x + self.rel_pos_embed[:, rel_pos_idx].reshape(1, *x.shape[1:])


# Extracts windows from an input tensor
def window_partition(x, window_shape):
    B, T, H, W, C = x.shape
    x = x.reshape(B, T // window_shape[0], window_shape[0], H // window_shape[1], window_shape[1], W // window_shape[2], window_shape[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, utils.shape2size(window_shape), C)
    return windows

# Recomposes window-partitioned tensor into original tensor
def window_unpartition(input_shape, windows, window_shape):
    B, T, H, W, _ = input_shape
    x = windows.reshape(B, T // window_shape[0], H // window_shape[1], W // window_shape[2], window_shape[0], window_shape[1], window_shape[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, T, H, W, -1)
    return x

# Window size should not be larger than input size. If this is the case, adapts window size to input size.
# Optionally works with shift indexes as well.
def bound_window_shape(input_shape, window_shape, shift_shape=None):
    window_shape = list(window_shape)
    if shift_shape is not None: shift_shape = list(shift_shape)
    for i in range(len(input_shape)):
        if input_shape[i] <= window_shape[i]:
            window_shape[i] = input_shape[i]
            if shift_shape is not None: shift_shape[i] = 0

    if shift_shape is None:
        return tuple(window_shape)

    return tuple(window_shape), tuple(shift_shape)

# Computes an attention mask that assigns a low attention score (-100) to elements that appear to be neighboring after
# a roll operation, but are in fact far apart, hence attentional aggregation should be prevented.
def compute_mask(input_shape, window_shape, shift_shape, device):
    T, H, W = input_shape
    img_mask = torch.zeros((1, T, H, W, 1), device=device)
    cnt = 0
    for t in slice(-window_shape[0]), slice(-window_shape[0], -shift_shape[0]), slice(-shift_shape[0], None):
        for h in slice(-window_shape[1]), slice(-window_shape[1], -shift_shape[1]), slice(-shift_shape[1], None):
            for w in slice(-window_shape[2]), slice(-window_shape[2], -shift_shape[2]), slice(-shift_shape[2], None):
                img_mask[:, t, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_shape).reshape(-1, utils.shape2size(window_shape))
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask[attn_mask != 0] = -100.
    attn_mask[attn_mask == 0] = 0.
    return attn_mask

# Extracts windows from input tensor, and applies attention locally on each window. If necessary, pads and shifts the
# input before window partitioning
def window_attend(x, window_shape, shift_shape, attn, mask_matrix):
    B, T, H, W, C = x.shape
    input_shape = [T, H, W]
    window_shape, shift_shape = bound_window_shape(input_shape, window_shape, shift_shape)

    # pad feature maps to multiples of window size
    pad = [(-input_shape[i]) % window_shape[i] for i in range(len(window_shape))]
    x = F.pad(x, [0, 0, 0, pad[2], 0, pad[1], 0, pad[0]])

    # cyclic shift
    shifted_x = x
    attn_mask = None
    if any(i > 0 for i in shift_shape):
        shifted_x = torch.roll(x, shifts=(-shift_shape[0], -shift_shape[1], -shift_shape[2]), dims=(1, 2, 3))
        attn_mask = mask_matrix

    # partition windows
    windows = window_partition(shifted_x, window_shape)
    # attention
    windows = attn(windows, bound_window_shape=window_shape, mask=attn_mask)
    # unpartition windows
    shifted_x = window_unpartition(x.shape, windows, window_shape)

    # reverse cyclic shift
    x = shifted_x
    if any(i > 0 for i in shift_shape):
        x = torch.roll(shifted_x, shifts=shift_shape, dims=(1, 2, 3))

    # Unpad
    x = x[:, :T, :H, :W, :]

    return x

# A swin transformer block applies local window attention on the input, followed by MLP token mapping
class SwinTransformer3DBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(self.dim, window_shape=self.window_size, num_heads=self.num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        x = x + self.drop_path(window_attend(self.norm1(x), self.window_size, self.shift_size, self.attn, mask_matrix))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# A sequence of Swin Transformer Blocks. Each block applies attention on local windows extracted from the input.
# In the next block, the input is shifted by half a window, and windows are again extracted from the shifted positions
# for local attention processing. This allows to aggregate information also across adjacent windows.
class SwinTransformer3DStage(nn.Module):
    def __init__(self,
                 dim,
                 out_dim,
                 depth,
                 num_heads,
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = [w // 2 for w in window_size]
        self.depth = depth
        self.num_heads = num_heads

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformer3DBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    shift_size=[0, 0, 0] if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                ))

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer)

    def forward(self, x):
        B, T, H, W, C = x.shape
        input_shape = [T, H, W]
        window_size, shift_size = bound_window_shape(input_shape, self.window_size, self.shift_size)
        padded_input_shape = [input_shape[i] + (-input_shape[i] % window_size[i]) for i in range(len(input_shape))]
        attn_mask = compute_mask(padded_input_shape, window_size, shift_size, x.device)

        for block in self.blocks:
            x = block(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


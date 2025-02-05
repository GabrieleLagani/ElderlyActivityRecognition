import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

import utils


def channel_split(x, splits):
	if x.shape[1] != sum(splits):
		raise RuntimeError("Tensor with {} channels cannot be divided into splits of sizes {}".format(x.shape[1], splits))
	split_idx = [0]
	for i in range(len(splits)): split_idx.append(split_idx[-1] + splits[i])
	return (x[:, split_idx[i]:split_idx[i + 1]] for i in range(len(splits)))

def stable_softmax(x, dim=None, ub=1e4):
	m = torch.max(x, dim=dim, keepdim=True)[0]
	n = x.numel() if dim is None else utils.shape2size( [x.shape[d] for d in (dim if isinstance(dim, (list, tuple)) else [dim])] )
	c = m - torch.log(torch.tensor(ub)/n)
	return (x - c).softmax(dim=dim)

def qkv_aggregate(q, k, v):
	scale = v.shape[-1] ** -0.5
	attn = (q*scale) @ (k.transpose(-2, -1)*scale)
	attn = attn.softmax(dim=-1)
	#attn = stable_softmax(attn, dim=-1)
	res = (attn @ v)
	return res

def linear_qkv_aggregate(q, k, v, eps=1e-5):
	d = q @ (k.transpose(-2, -1).sum(dim=-1, keepdim=True))
	return (q @ (k.transpose(-2, -1) @ v)) / (d + eps)

def split_heads(x, headsize):
	B, N, C = x.shape
	if C % headsize != 0:
		raise RuntimeError("Head size ({}) does not divide total number of available channels ({}) for head splitting".format(headsize, C))
	return x.reshape(B, N, headsize, -1).permute(0, 3, 1, 2)

def merge_heads(x):
	B, H, N, C = x.shape
	return x.permute(0, 2, 3, 1).reshape(B, N, C*H)

def merge_heads_and_channels(x):
	return x.transpose(0, -1).reshape(x.shape[-1]*x.shape[1], *x.shape[2:-1], x.shape[0]).transpose(0, -1)

def restore_heads_and_channels(x, headsize):
	return x.transpose(0, -1).reshape(headsize, -1, *x.shape[1:-1], x.shape[0]).transpose(0, -1)

def normalize_ksp3d(kernel_size, stride, padding, token_dim=None, transposed=False):
	token_dim = token_dim if token_dim is None or isinstance(token_dim, (list, tuple)) else [token_dim]
	kernel_size = [(k if (token_dim is None or (i + 2) in token_dim) else 1) for i, k in enumerate(_triple(kernel_size))]
	stride = [(k if (token_dim is None or (i + 2) in token_dim) else 1) for i, k in enumerate(_triple(stride))]
	if padding == 'same': padding = 0 if transposed else utils.get_padding_same(kernel_size)
	padding = [(k if (token_dim is None or (i + 2) in token_dim) else 0) for i, k in enumerate(_triple(padding))]
	return kernel_size, stride, padding

def unfold(x, kernel_size=3, stride=1, padding=0):
	kernel_size = _triple(kernel_size)
	stride = _triple(stride)
	B, C, T, H, W = x.shape
	x = torch.conv3d(x.reshape(-1, 1, T, H, W),
					 torch.eye(utils.shape2size(kernel_size),
							   device=x.device,
							   dtype=x.dtype).reshape(utils.shape2size(kernel_size), 1, *kernel_size),
					 padding=padding,
					 stride=stride)
	return x.reshape(B, -1, *x.shape[2:])

def fold(x, kernel_size=3, stride=1):
	kernel_size = _triple(kernel_size)
	stride = _triple(stride)
	B, C, T, H, W = x.shape
	x = torch.conv_transpose3d(x.reshape(-1, utils.shape2size(kernel_size), T, H, W),
				  torch.eye(utils.shape2size(kernel_size),
							device=x.device,
							dtype=x.dtype).reshape(utils.shape2size(kernel_size), 1, *kernel_size),
				  stride=stride)
	return x.reshape(B, -1, *x.shape[2:])

def gauss(x):
	return torch.exp(-(x**2).sum(dim=-1, keepdim=True))

def get_3d_sincos_pos_embed(embed_dim, shape):
	embed_dim_spatial = 2 * embed_dim // 3
	embed_dim_temporal = embed_dim - embed_dim_spatial

	# spatial
	grid_h_size = shape[1]
	grid_h = torch.arange(grid_h_size).float() / grid_h_size

	grid_w_size = shape[2]
	grid_w = torch.arange(grid_w_size).float() / grid_w_size

	grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
	grid = torch.stack(grid, dim=0)
	grid = grid.reshape([2, 1, grid_h_size, grid_w_size])

	emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_spatial // 2, grid[0])
	emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_spatial // 2, grid[1])
	pos_embed_spatial = torch.cat([emb_h, emb_w], dim=1)

	# temporal
	t_size = shape[0]
	grid_t = torch.arange(t_size).float() / t_size
	pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

	pos_embed_temporal = torch.repeat_interleave(pos_embed_temporal.unsqueeze(1), grid_h_size * grid_w_size, dim=1)
	pos_embed_spatial = torch.repeat_interleave(pos_embed_spatial.unsqueeze(0), t_size, dim=0)

	pos_embed = torch.cat([pos_embed_temporal, pos_embed_spatial], dim=-1)
	pos_embed = pos_embed.reshape(-1, embed_dim)

	return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, base=10000.):
	omega = torch.arange(embed_dim // 2).float()
	omega /= embed_dim / 2.0
	omega = 1.0 / base**omega

	pos = pos.reshape(-1)
	out = pos.unsqueeze(1) * omega.unsqueeze(0)

	emb_sin = torch.sin(out)
	emb_cos = torch.cos(out)

	emb = torch.cat([emb_sin, emb_cos], dim=1)
	return emb


class Tokenizer:
	def __init__(self, all_dim, token_dim=None):
		token_dim = token_dim if token_dim is None or isinstance(token_dim, (list, tuple)) else [token_dim]
		self.shape = []
		self.inv_perm = []
		self.all_dim = all_dim # dim 0 is the batch dimension, dim 1 is the channel dimension. All other dims (from 2 onward) are time, height, width...
		self.token_dim = [self.all_dim[d] for d in (self.all_dim[2:] if token_dim is None else token_dim)]
		if 0 in self.token_dim or 1 in self.token_dim:
			raise ValueError("Token dimensions ({}) can't be the batch (0) or channel (1) dimension".format(token_dim))
		self.batch_dim = [d for d in self.all_dim if d not in [0, 1] + self.token_dim]

	def get_inv_perm(self):
		inv_perm = {d: i for i, d in enumerate(self.batch_dim + self.token_dim)}
		inv_perm = [inv_perm[d+2] for d in range(len(inv_perm))]
		#inv_perm = sorted((d, i) for i, d in enumerate(batch_dim + token_dim))
		#inv_perm = [inv_perm[d][1] for d in range(len(inv_perm))]
		return inv_perm

	def tokenize(self, x):
		self.shape = list(x.shape)
		self.inv_perm = self.get_inv_perm()
		B, N, C = utils.shape2size([self.shape[d] for d in (self.batch_dim + [0])]), utils.shape2size([self.shape[d] for d in self.token_dim]), self.shape[1]
		x = x.permute(0, *self.batch_dim, *self.token_dim, 1).reshape(B, N, C)
		return x

	def untokenize(self, x):
		x = x.reshape(self.shape[0],
					  *(self.shape[d] for d in self.batch_dim),
					  *(self.shape[d] for d in self.token_dim),
					  -1).permute(0, -1, *(i + 1 for i in self.inv_perm))
		return x


class LayerNorm(nn.Module):
	def __init__(self, features):
		super().__init__()
		self.features = features
		self.norm = nn.LayerNorm(features)

	def forward(self, x):
		return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

class FullLayerNorm(nn.Module):
	def __init__(self, features):
		super().__init__()
		self.features = features
		self.norm = None

	def forward(self, x):
		if self.norm is None: # Lazy init
			self.norm = nn.LayerNorm(utils.shape2size(x.shape[1:]))
		return self.norm(x.reshape(x.shape[0], -1)).reshape(*x.shape)

class BatchNorm(nn.Module):
	def __init__(self, features):
		super().__init__()
		self.features = features
		self.norm = nn.BatchNorm3d(features)

	def forward(self, x):
		return self.norm(x)

class InstanceNorm(nn.Module):
	def __init__(self, features):
		super().__init__()
		self.features = features
		self.norm = nn.InstanceNorm3d(features, affine=True)

	def forward(self, x):
		return self.norm(x)

class LocalContrastNorm(nn.Module):
	def __init__(self, features):
		super().__init__()
		self.features = features
		self.norm = nn.LocalResponseNorm(features)

	def forward(self, x):
		return self.norm(x)


class TokenAggregator(nn.Module):
	def __init__(self, size=64, features=64, heads=8, shared_map=False, transposed=False):
		super().__init__()
		self.size = size
		self.features = features
		self.heads = heads
		self.shared_map = shared_map
		self.transposed = transposed
		num_banks = 1 if self.shared_map else self.heads
		q = nn.init.xavier_normal_(torch.empty((num_banks * self.size, self.features))).reshape(num_banks, self.size, self.features)
		self.q = nn.Parameter(q, requires_grad=True)

	def new(self):
		return TokenAggregator(self.size, self.features, self.heads, shared_map=self.shared_map, transposed=self.transposed)

	def forward(self, k, v):
		if self.transposed: k, v = k.transpose(-2, -1), v.transpose(-2, -1)
		x = qkv_aggregate(self.q.unsqueeze(0), k, v)
		if self.transposed: x = x.transpose(-2, -1)
		return x


class SplitIdentity(nn.Module):
	def __init__(self, splits=1):
		super().__init__()
		self.splits = splits

	def new(self):
		return SplitIdentity(splits=self.splits)

	def forward(self, x:torch.Tensor):
		return x.repeat((1, self.splits, 1, 1, 1))

class MultiHeadLinear(nn.Module):
	def __init__(self, in_features=512, out_features=512, heads=1, headsize=512, shared_map=False, bias=True,
				 token_dim=None, transposed=False):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.heads = heads
		self.headsize = headsize if transposed else self.in_features
		self.shared_map = shared_map
		self.token_dim = token_dim if token_dim is None or isinstance(token_dim, (list, tuple)) else [token_dim]
		self.transposed = transposed
		if self.transposed and self.out_features % self.in_features != 0:
			raise ValueError("Output features ({}) must be multiple of input features ({}) in transposed mode".format(self.out_features, self.in_features))
		self.use_bias = bias

		self.tokenizer = None
		self.init_filters()

	def init_filters(self):
		num_banks = 1 if self.shared_map else self.heads
		w = nn.init.xavier_normal_(torch.empty((num_banks * self.out_features, self.in_features)))
		self.weight = nn.Parameter(w.reshape(num_banks, self.out_features, self.in_features), requires_grad=True)
		b = nn.init.zeros_(torch.empty((num_banks, self.out_features)))
		self.bias = nn.Parameter(b, requires_grad=True) if self.use_bias else None

	def new(self):
		return MultiHeadLinear(self.in_features, self.out_features, self.heads, self.headsize, shared_map=self.shared_map,
							   bias=self.use_bias, token_dim=self.token_dim, transposed=self.transposed)

	def apply_filters(self, x):
		x = x.matmul(self.weight.unsqueeze(0).transpose(-2, -1))
		if self.use_bias: x = x + self.bias.unsqueeze(0).unsqueeze(-2)
		return x

	def forward_tokens(self, x):
		B, H, N, C = x.shape
		if self.transposed: x = x.transpose(-2, -1)
		x = self.apply_filters(x)
		if self.transposed: x = x.reshape(B, -1, C, self.out_features // N, N).transpose(-3, -1).reshape(B, -1, N, (self.out_features // N) * C)
		else: x = x.reshape(B, -1, N, self.out_features)
		return x

	def forward(self, x):
		self.tokenizer = Tokenizer(list(range(x.ndim)), self.token_dim)
		x = split_heads(self.tokenizer.tokenize(x), self.headsize)
		x = self.forward_tokens(x)
		return self.tokenizer.untokenize(merge_heads(x))

class MultiHeadConv3d(MultiHeadLinear):
	def __init__(self, in_channels=512, out_channels=512, heads=1, headsize=512, shared_map=False, bias=True,
				 kernel_size=3, stride=1, padding=0, token_dim=None, transposed=False):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size, self.stride, self.padding = normalize_ksp3d(kernel_size, stride, padding, token_dim, transposed)
		in_features = self.in_channels if transposed else self.in_channels * utils.shape2size(self.kernel_size)
		out_features = self.out_channels
		super().__init__(in_features, out_features, heads, headsize*utils.shape2size(self.kernel_size), shared_map=shared_map,
						 bias=bias, token_dim=token_dim, transposed=transposed)
		self.headsize = headsize if self.transposed else self.in_channels

	def new(self):
		return MultiHeadConv3d(self.in_channels, self.out_channels, self.heads, self.headsize, shared_map=self.shared_map,
							   bias=self.use_bias, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
							   token_dim=self.token_dim, transposed=self.transposed)

	def apply_conv_filters(self, x):
		w = self.weight.reshape(self.weight.shape[0] * self.weight.shape[1], self.in_channels, *self.kernel_size)
		b = self.bias.reshape(-1) if self.use_bias else None
		if self.shared_map:
			out = torch.conv3d(x.reshape(x.shape[0], self.headsize, -1, *x.shape[2:]).transpose(1, 2).reshape(-1, self.headsize, *x.shape[2:]),
							   w, bias=b, stride=self.stride, padding=self.padding)
			out = out.reshape(x.shape[0], -1, out.shape[1], *out.shape[2:]).transpose(1, 2).reshape(x.shape[0], -1, *out.shape[2:])
		else:
			if (self.heads * self.headsize) % x.shape[1] != 0:
				raise RuntimeError("Expected number of channels ({}) is not divisible by input channels ({})".format(self.heads * self.headsize, x.shape[1]))
			x = x.repeat_interleave((self.heads * self.headsize) // x.shape[1], dim=1)
			out = torch.conv3d(x.reshape(x.shape[0], self.headsize, -1, *x.shape[2:]).transpose(1, 2).reshape(x.shape[0], -1, *x.shape[2:]),
							   w, bias=b, stride=self.stride, padding=self.padding, groups=self.heads)
			out = out.reshape(x.shape[0], -1, self.weight.shape[1], *out.shape[2:]).transpose(1, 2).reshape(x.shape[0], -1, *out.shape[2:])
		return out

	def forward(self, x):
		#x = unfold(x, kernel_size=_triple(self.kernel_size), stride=self.stride, padding=self.padding)
		#return super().forward(x)
		if self.transposed:
			x = unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
			self.headsize = self.headsize * utils.shape2size(self.kernel_size)
			x = super().forward(x)
			self.headsize = self.headsize // utils.shape2size(self.kernel_size)
			return x
		return self.apply_conv_filters(x)

class MultiHeadConv2Plus1d(MultiHeadConv3d):
	def __init__(self, in_channels=512, out_channels=512, heads=1, headsize=512, shared_map=False, bias=True,
				 kernel_size=3, stride=1, padding=0, token_dim=None, transposed=False):
		nn.Module.__init__(self)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.heads = heads
		self.headsize = headsize
		self.shared_map = shared_map
		self.use_bias = bias
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.token_dim = token_dim
		self.transposed = transposed

		kernel_size, stride, padding = _triple(kernel_size), _triple(stride), _triple(padding) if isinstance(padding, int) else padding
		full_size = kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels
		split_size = kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels
		#interm_channels = in_channels if transposed else full_size // split_size # More channels, but more space occupancy
		interm_channels = in_channels # Fewer channels, but more lightweight
		self.spatial_conv = MultiHeadConv3d(in_channels=in_channels, out_channels=interm_channels, heads=heads, headsize=headsize, shared_map=shared_map, bias=bias,
				 kernel_size=(1, kernel_size[1], kernel_size[2]), stride=(1, stride[1], stride[2]), padding=padding if isinstance(padding, str) else (0, padding[1], padding[2]), token_dim=token_dim, transposed=transposed)
		self.temporal_conv = MultiHeadConv3d(in_channels=interm_channels, out_channels=out_channels, heads=heads, headsize=headsize, shared_map=shared_map, bias=bias,
				 kernel_size=(kernel_size[0], 1, 1), stride=(stride[0], 1, 1), padding=padding if isinstance(padding, str) else (padding[0], 1, 2), token_dim=token_dim, transposed=transposed)

	def new(self):
		return MultiHeadConv2Plus1d(self.in_channels, self.out_channels, self.heads, self.headsize, shared_map=self.shared_map,
							   bias=self.use_bias, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
							   token_dim=self.token_dim, transposed=self.transposed)

	def forward(self, x):
		return self.temporal_conv(self.spatial_conv(x))

class PosEmbedding(nn.Module):
	def __init__(self, size, channels=64, heads=8, shared_map=False, token_dim=None, transposed=False, cape_reg=0):
		super().__init__()
		self.size = size
		self.channels = channels
		self.heads = heads
		self.headsize = self.channels
		self.shared_map = shared_map
		self.token_dim = token_dim if token_dim is None or isinstance(token_dim, (list, tuple)) else [token_dim]
		self.transposed = transposed

		self.tokenizer = None

		# Collapse avoiding position embedding regularizer
		self.cape_reg = cape_reg
		self.cape_reg_buff = 0

		self.init_filters()

	def init_filters(self):
		num_banks = 1 if self.shared_map else self.heads
		pos_embed = nn.init.xavier_normal_(torch.empty((num_banks * self.channels, self.size)))
		self.pos_embed = nn.Parameter(pos_embed, requires_grad=True)

	def new(self):
		return PosEmbedding(self.size, self.heads, self.channels, shared_map=self.shared_map, token_dim=self.token_dim, transposed=self.transposed)

	def apply_pos_embed(self, x):
		B, H, N, C = x.shape
		return x + self.pos_embed.reshape(1, -1, C, N).transpose(-2, -1)

	def forward(self, x):
		self.tokenizer = Tokenizer(list(range(x.ndim)), token_dim=self.token_dim)
		x = split_heads(self.tokenizer.tokenize(x), self.headsize)

		pos_emb = self.apply_pos_embed(x)

		if self.training and self.cape_reg > 0:
			# Compute and store collision avoiding position embedding regularizer
			token_dim = 3 if self.transposed else 2
			idx = torch.randperm(x.shape[token_dim], device=x.device)
			inv_idx = torch.arange(x.shape[token_dim], device=x.device)[idx]
			def permute_tokens(x, idx):
				return x[:, :, :, idx] if self.transposed else x[:, :, idx, :]
			self.cape_reg_buff = -self.cape_reg * (torch.norm(
				pos_emb.clone().detach() - permute_tokens(self.apply_pos_embed(permute_tokens(x.clone().detach(), idx)), inv_idx),
				p=2, dim=token_dim)**2).sum()
		else: self.cape_reg_buff = 0

		return self.tokenizer.untokenize(merge_heads(pos_emb))

class SinCosPosEmbedding(PosEmbedding):
	def __init__(self, shape, channels=64, heads=8, shared_map=False, token_dim=None, transposed=False, cape_reg=0):
		self.shape = shape
		super().__init__(size=shape, channels=channels, heads=heads, shared_map=shared_map, token_dim=token_dim, transposed=transposed, cape_reg=cape_reg)

	def init_filters(self):
		num_banks = 1 if self.shared_map else self.heads
		pos_embed = get_3d_sincos_pos_embed(
			embed_dim=num_banks * self.channels,
			shape=self.shape
		)
		self.pos_embed = nn.Parameter(pos_embed, requires_grad=True)

class AppendedPosEmbedding(PosEmbedding):
	def apply_pos_embed(self, x):
		B, H, N, C = x.shape
		return torch.cat([x, self.pos_embed.reshape(1, -1, C, N).transpose(-2, -1)], dim=-1)

class MultiplicativePosEmbedding(PosEmbedding):
	def __init__(self, size, channels=64, heads=8, shared_map=False, token_dim=None, transposed=False, cape_reg=0):
		super().__init__(size, channels, heads, shared_map=shared_map, token_dim=token_dim, transposed=transposed, cape_reg=cape_reg)
		self.w_pos_embed = nn.Parameter(1 + nn.init.xavier_normal_(torch.empty_like(self.pos_embed)), requires_grad=True)

	def new(self):
		return MultiplicativePosEmbedding(self.size, self.heads, self.channels, shared_map=self.shared_map, token_dim=self.token_dim, transposed=self.transposed)

	def apply_pos_embed(self, x):
		B, H, N, C = x.shape
		return super().apply_pos_embed(x * self.w_pos_embed.reshape(1, -1, C, N).transpose(-2, -1))

class AffinePosEmbedding(PosEmbedding):
	def __init__(self, size, in_features=64, out_features=64, heads=8, shared_map=False, token_dim=None, transposed=False, cape_reg=0):
		self.in_features = in_features
		self.out_features = out_features
		if transposed and self.out_features % self.in_features != 0:
			raise ValueError("Output features ({}) must be multiple of input features ({}) in transposed mode".format(self.out_features, self.in_features))
		super().__init__(size, self.out_features, heads, shared_map=shared_map, token_dim=token_dim, transposed=transposed, cape_reg=cape_reg)
		self.headsize = self.size if self.transposed else self.in_features

	def init_filters(self):
		num_banks = 1 if self.shared_map else self.heads
		aff_pos_embed = nn.init.xavier_normal_(torch.empty((num_banks * self.out_features, self.in_features, self.size)))
		self.aff_pos_embed = nn.Parameter(aff_pos_embed.reshape(-1, self.out_features, self.in_features, self.size).permute(0, 3, 2, 1), requires_grad=True)
		super().init_filters()

	def new(self):
		return AffinePosEmbedding(self.size, self.in_features, self.out_features, self.heads, shared_map=self.shared_map,
								  token_dim=self.token_dim, transposed=self.transposed, cape_reg=self.cape_reg)

	def apply_pos_embed(self, x):
		B, H, N, C = x.shape
		if self.transposed: x = x.transpose(-2, -1)
		x = x.unsqueeze(-2).matmul(self.aff_pos_embed.unsqueeze(0)).reshape(B, -1, self.aff_pos_embed.shape[1], self.aff_pos_embed.shape[-1])
		x = super().apply_pos_embed(x)
		if self.transposed: x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, self.in_features).transpose(-2, -3).reshape(x.shape[0], x.shape[1], -1, self.in_features).transpose(-2, -1)
		return x

class NonlinearPosEmbedding(AffinePosEmbedding):
	def __init__(self, size, in_features=64, out_features=64, heads=8, shared_map=False, nonlin=None, token_dim=None, transposed=False, cape_reg=0):
		self.nonlin = nonlin
		super().__init__(size, in_features, out_features, heads, shared_map=shared_map, token_dim=token_dim, transposed=transposed, cape_reg=cape_reg)

	def init_filters(self):
		nonlin_heads = self.size if self.shared_map else self.heads * self.size
		self.nonlin = self.nonlin if self.nonlin is not None else MultiHeadMLP(
			in_features=self.in_features, hidden_features=self.out_features, out_features=self.out_features,
			heads=nonlin_heads, shared_map=False, token_dim=self.token_dim)

	def new(self):
		return NonlinearPosEmbedding(self.size, self.in_features, self.out_features, self.heads, shared_map=self.shared_map,
									 nonlin=self.nonlin.new(), token_dim=self.token_dim, transposed=self.transposed, cape_reg=self.cape_reg)

	def apply_pos_embed(self, x):
		B, H, N, C = x.shape
		if self.transposed: x.transpose(-2, -1)
		x = x.reshape(B * H, N * C, 1, 1, 1) if self.shared_map else x.reshape(B, H * N * C, 1, 1, 1)
		x = self.nonlin(x).reshape(B, -1, self.size, self.out_features)
		if self.transposed: x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, self.in_features).transpose(-2, -3).reshape(x.shape[0], x.shape[1], -1, self.in_features).transpose(-2, -1)
		return x

class RotaryPosEmbedding(nn.Module):
	def __init__(self, dim, n=4096, base=10000):
		super().__init__()

		self.dim = dim
		self.base = base
		self.n = n

		self.build_rope(self.n)

	def build_rope(self, n):

		# Create theta values as 1/base^k for k = 0/dim, 2/dim, ... dim/dim=1
		theta = 1.0 / ( self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim) )

		# Create position indexes `[0, 1, ..., max_seq_len - 1]`
		seq_idx = torch.arange(n, dtype=theta.dtype, device=theta.device)

		# Outer product of theta and position index; output tensor has a shape of [max_seq_len, dim // 2]
		idx_theta = (seq_idx.unsqueeze(1) * theta.unsqueeze(0)).float()

		# rope includes both the cos and sin components and so the output shape is [max_seq_len, dim // 2, 2]
		rope = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
		self.rope = nn.Parameter(rope, requires_grad=True)

	def forward(self, x):
		seq_len = x.shape[1]
		xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
		rope = self.rope.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
		x_out = torch.stack([
				xshaped[..., 0] * rope[..., 0] - xshaped[..., 1] * rope[..., 1],
				xshaped[..., 1] * rope[..., 0] + xshaped[..., 0] * rope[..., 1],
			], dim=-1)

		x_out = x_out.flatten(3)
		return x_out.type_as(x)

class RelativePosEmbedding(nn.Module):
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

class Attention(nn.Module):
	QKV_SPLITS = 3

	def __init__(self, channels=64, heads=8, token_dim=None, transposed=False, qkv=None, qkv_splits=None, proj=None, attn_drop=0., proj_drop=0.):
		super().__init__()

		self.token_dim = token_dim if token_dim is None or isinstance(token_dim, (list, tuple)) else [token_dim]

		self.channels = channels
		self.heads = heads
		self.transposed = transposed

		self.tokenizer = None
		self.qkv_splits = qkv_splits if qkv_splits is not None else [self.channels * self.heads for _ in range(self.QKV_SPLITS)]
		if len(self.qkv_splits) != self.QKV_SPLITS: raise ValueError("Expected {} splits for QKV mapping, but {} were found".format(self.QKV_SPLITS, len(self.qkv_splits)))
		self.qkv = qkv if qkv is not None else SplitIdentity(self.QKV_SPLITS)

		self.proj = proj if proj is not None else SplitIdentity()
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj_drop = nn.Dropout(proj_drop)

	def new(self):
		return Attention(self.channels, self.heads, token_dim=self.token_dim, transposed=self.transposed,
						 qkv=self.qkv.new(), qkv_splits=self.qkv_splits,
						 proj=self.proj.new(), attn_drop=self.attn_drop.p, proj_drop=self.proj_drop.p)

	def aggregate(self, qkv):
		q, k, v = qkv
		return qkv_aggregate(q, k, v)

	def attend(self, qkv):
		if self.transposed: qkv = (s.transpose(-2, -1) for s in qkv)
		x = self.aggregate(qkv)
		if self.transposed: x = x.transpose(-2, -1)
		return x

	def forward(self, x):
		self.tokenizer = Tokenizer(list(range(x.ndim)), self.token_dim)
		qkv = channel_split(self.qkv(x), self.qkv_splits)
		qkv = (split_heads(self.tokenizer.tokenize(s), self.channels) for s in qkv)
		x = self.attend(qkv)
		x = self.tokenizer.untokenize(merge_heads(x))
		x = self.proj_drop(self.proj(self.attn_drop(x)))
		return x

class AttentionWithRelPosEmb(Attention):
	def __init__(self):
		raise NotImplemented

class KernelAttention(Attention):
	def __init__(self, channels=64, heads=8, features=64, knl_size=64, token_dim=None, transposed=False, qkv=None, qkv_splits=None,
				 proj=None, attn_drop=0., proj_drop=0.):
		super().__init__(channels, heads, token_dim=token_dim, transposed=transposed, qkv=qkv, qkv_splits=qkv_splits,
						 proj=proj, attn_drop=attn_drop, proj_drop=proj_drop)
		self.features = features if self.transposed else self.channels
		self.knl_size = knl_size
		r = nn.init.xavier_normal_(torch.empty(self.knl_size, self.features))
		self.r = nn.Parameter(r, requires_grad=True)

	def new(self):
		return KernelAttention(self.channels, self.heads, features=self.features, knl_size=self.knl_size,
							   token_dim=self.token_dim, transposed=self.transposed, qkv=self.qkv, qkv_splits=self.qkv_splits,
							   proj=self.proj.new(), attn_drop=self.attn_drop.p, proj_drop=self.proj_drop.p)

	def aggregate(self, qkv):
		q, k, v = qkv
		scale = q.shape[-1] ** -0.5
		q, k = q * scale, k * scale
		hq, hk = gauss(q) / (self.knl_size**0.5), gauss(k) / (self.knl_size**0.5)
		return linear_qkv_aggregate(hq * torch.exp(q.matmul(self.r.t())), hk * torch.exp(k.matmul(self.r.t())), v)

class HiddenRankAttention(Attention):
	QKV_SPLITS = 5

	def __init__(self, channels=64, heads=8, features=64, hidden_rank=64, token_dim=None, transposed=False, qkv=None, qkv_splits=None,
				 proj=None, attn_drop=0., proj_drop=0.):
		super().__init__(channels, heads, token_dim=token_dim, transposed=transposed, qkv=qkv, qkv_splits=qkv_splits,
						 proj=proj, attn_drop=attn_drop, proj_drop=proj_drop)
		self.features = features if self.transposed else channels
		self.hidden_rank = hidden_rank
		self.k_aggr = TokenAggregator(self.hidden_rank, self.features, self.heads, shared_map=True)
		self.v_aggr = TokenAggregator(self.hidden_rank, self.features, self.heads, shared_map=True)

	def new(self):
		return HiddenRankAttention(self.channels, self.heads,
								   features=self.features, hidden_rank=self.hidden_rank,
								   token_dim=self.token_dim, transposed=self.transposed,
								   qkv=self.qkv.new(), qkv_splits=self.qkv_splits,
								   proj=self.proj.new(), attn_drop=self.attn_drop.p, proj_drop=self.proj_drop.p)

	def aggregate(self, qkv):
		q, k_k, k_v, v_k, v_v = qkv
		return qkv_aggregate(q, self.k_aggr(k_k, k_v), self.v_aggr(v_k, v_v))

class LocalAttention(Attention):
	def __init__(self, channels=64, heads=8, kernel_size=3, stride=1, padding=0, token_dim=None, transposed=False, refold=True,
				 qkv=None, qkv_splits=None, proj=None, attn_drop=0., proj_drop=0.):
		self.kernel_size, self.stride, self.padding = normalize_ksp3d(kernel_size, stride, padding, token_dim, transposed)
		self.refold = refold
		super().__init__(channels, heads, token_dim=token_dim, transposed=transposed, qkv=qkv, qkv_splits=qkv_splits,
						 proj=proj, attn_drop=attn_drop, proj_drop=proj_drop)

	def new(self):
		return LocalAttention(self.channels, self.heads, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
							  token_dim=self.token_dim, transposed=self.transposed, refold=self.refold,
							  qkv=self.qkv.new(), qkv_splits=self.qkv_splits,
							  proj=self.proj.new(), attn_drop=self.attn_drop.p, proj_drop=self.proj_drop.p)

	def attend(self, qkv):
		q, k, v = qkv
		k, v = self.tokenizer.untokenize(merge_heads(k)), self.tokenizer.untokenize(merge_heads(v))
		q = self.tokenizer.untokenize(merge_heads(q))
		kernel_size = [k if k > 0 else q.shape[i+2] for i, k in enumerate(self.kernel_size)]
		stride = [k if k > 0 else 1 for i, k in enumerate(self.stride)]
		k = unfold(k, kernel_size=kernel_size, stride=stride, padding=self.padding)
		v = unfold(v, kernel_size=kernel_size, stride=stride, padding=self.padding)
		if self.transposed: q = unfold(q, kernel_size=kernel_size, stride=stride, padding=self.padding)
		else:
			padding = [k if k > 0 else 0 for i, k in enumerate(self.padding)]
			padding = [padding[-1 - i//2] for i in range(2*len(padding))]
			q = F.pad(q, padding)
			start = [(kernel_size[i] - 1) // 2 for i in range(len(kernel_size))]
			end = [q.shape[2 + i] - start[i] for i in range(len(start))]
			q = q[:, :, start[0]:end[0]:stride[0], start[1]:end[1]:stride[1], start[2]:end[2]:stride[2]]
		k, v = split_heads(self.tokenizer.tokenize(k), self.channels * utils.shape2size(kernel_size)), split_heads(self.tokenizer.tokenize(v), self.channels * utils.shape2size(kernel_size))
		if self.transposed: q = split_heads(self.tokenizer.tokenize(q), self.channels * utils.shape2size(kernel_size))
		else: q = split_heads(self.tokenizer.tokenize(q), self.channels)
		q, k, v = (s.transpose(1, 2).reshape(s.shape[0], s.shape[2], s.shape[1], self.channels, -1).transpose(-2, -1) for s in (q, k, v))

		x = super().attend((q, k, v))
		x = x.transpose(1, 2).transpose(-2, -1).reshape(x.shape[0], x.shape[2], x.shape[1], -1)

		if self.transposed and self.refold:
			x = self.tokenizer.untokenize(merge_heads(x))
			x = fold(x, kernel_size=kernel_size, stride=stride)
			x = split_heads(self.tokenizer.tokenize(x), self.channels)
		return x


class AttentiveBias(nn.Module):
	QKV_SPLITS = 2

	def __init__(self, channels=64, heads=8, headsize=64, shared_map=False, transposed=False, qkv=None, qkv_splits=None):
		super().__init__()
		self.channels = channels
		self.heads = heads
		self.headsize = headsize if transposed else self.channels
		self.shared_map = shared_map
		self.transposed = transposed

		self.tokenizer = None
		self.qkv_splits = qkv_splits if qkv_splits is not None else [self.headsize * self.heads for _ in range(self.QKV_SPLITS)]
		if len(self.qkv_splits) != self.QKV_SPLITS: raise ValueError("Expected {} splits for QKV mapping, but {} were found".format(self.QKV_SPLITS, len(self.qkv_splits)))
		self.qkv = qkv if qkv is not None else SplitIdentity(self.QKV_SPLITS)
		self.aggr = TokenAggregator(1, self.channels, self.heads, self.shared_map, self.transposed)

	def new(self):
		return AttentiveBias(self.channels, self.heads, self.headsize, shared_map=self.shared_map, transposed=self.transposed,
							 qkv=self.qkv.new(), qkv_splits=self.qkv_splits)

	def get_bias(self, x):
		qkv = channel_split(self.qkv(x), self.qkv_splits)
		k, v = qkv
		k, v = split_heads(self.tokenizer.tokenize(k), self.headsize), split_heads(self.tokenizer.tokenize(v), self.headsize)
		bias = self.aggr(k, v).transposed(-2, -1)
		if self.transposed: return bias.repeat([1, 1, self.headsize, 1]).transpose(1, 2).reshape(x.shape[0], -1, *x.shape[2:])
		return bias.transpose(1, 2).reshape(x.shape[0], -1, 1, 1, 1)

	def forward(self, x):
		self.tokenizer = Tokenizer(list(range(x.ndim)))
		return self.get_bias(x)

class AttentiveConv3d(MultiHeadConv3d):
	QKV_SPLITS = 2

	def __init__(self, in_channels=64, out_channels=64, heads=8, headsize=64, shared_map=False, bias=True,
				 kernel_size=3, stride=1, padding=0, token_dim=None, transposed=False,
				 qkv=None, qkv_splits=None, qkv_b=None, qkv_b_splits=None):
		self.qkv_splits = qkv_splits if qkv_splits is not None else [headsize * heads for _ in range(self.QKV_SPLITS)]
		if len(self.qkv_splits) != self.QKV_SPLITS: raise ValueError("Expected {} splits for QKV mapping, but {} were found".format(self.QKV_SPLITS, len(self.qkv_splits)))
		self.qkv = qkv if qkv is not None else SplitIdentity(self.QKV_SPLITS)
		self.qkv_b = qkv_b
		self.qkv_b_splits = qkv_b_splits
		super().__init__(in_channels, out_channels, heads, headsize, shared_map=shared_map, bias=bias,
						 kernel_size=kernel_size, stride=stride, padding=padding, token_dim=token_dim, transposed=transposed)

	def init_filters(self):
		channels = self.in_channels if self.transposed else self.in_channels * utils.shape2size(self.kernel_size)
		self.aggr = TokenAggregator(self.out_channels, channels, self.heads, self.shared_map, self.transposed)
		self.bias = AttentiveBias(self.out_channels, self.heads, self.headsize, shared_map=self.shared_map,
								  transposed=self.transposed, qkv=self.qkv_b, qkv_splits=self.qkv_b_splits) if self.use_bias else None

	def new(self):
		return AttentiveConv3d(self.in_channels, self.out_channels, self.heads, self.headsize, shared_map=self.shared_map,
							   bias=self.use_bias, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
							   token_dim=self.token_dim, transposed=self.transposed,
							   qkv=self.qkv.new(), qkv_splits=self.qkv_splits, qkv_b=self.qkv_b, qkv_b_splits=self.qkv_b_splits)

	def get_weight(self, x):
		qkv = channel_split(self.qkv(x), self.qkv_splits)
		k, v = (unfold(s, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding) for s in qkv)
		k, v = split_heads(self.tokenizer.tokenize(k), self.headsize * utils.shape2size(self.kernel_size)), split_heads(self.tokenizer.tokenize(v), self.headsize * utils.shape2size(self.kernel_size))
		weight = self.aggr(k, v)
		if self.transposed: return weight
		return weight.transpose(-2, -1)

	def forward(self, x):
		self.tokenizer = Tokenizer(list(range(x.ndim)))

		w = self.get_weight(x)
		x = unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
		x = split_heads(self.tokenizer.tokenize(x), self.headsize * utils.shape2size(self.kernel_size))
		B, H, N, C = x.shape
		if self.transposed: x.transpose(-2, -1)
		x = x.matmul(w)
		if self.transposed: x = x.transpose(-2, -1).reshape(B, -1, *self.kernel_size)
		else: x = self.tokenizer.untokenize(merge_heads(x))

		if self.use_bias: x = x + self.bias(x)

		return x

class AttentivePool3d(LocalAttention):
	QKV_SPLITS = 2

	def __init__(self, size=1, channels=64, heads=8, shared_map=False,
				 kernel_size=3, stride=1, padding=0, token_dim=None, transposed=False, refold=True,
				 qkv=None, qkv_splits=None, proj=None, attn_drop=0., proj_drop=0.):
		super().__init__(channels, heads, kernel_size=kernel_size, stride=stride, padding=padding,
						 token_dim=token_dim, transposed=transposed, refold=refold, qkv=qkv, qkv_splits=qkv_splits,
						 proj=proj, attn_drop=attn_drop, proj_drop=proj_drop)

		self.size = size
		if self.transposed and (any(k < 0 for k in self.kernel_size) or any(s < 0 for s in self.stride)):
			raise ValueError("Positive values expected for kernel size ({}) and stride ({}) in transposed mode".format(self.kernel_size, self.stride))
		self.features = utils.shape2size(self.kernel_size) if self.transposed else channels
		self.shared_map = shared_map
		self.aggr = TokenAggregator(self.size, self.features, self.heads, self.shared_map)

	def new(self):
		return AttentivePool3d(self.size, self.channels, self.heads, shared_map=self.shared_map,
							   kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
							   token_dim=self.token_dim, transposed=self.transposed, refold=self.refold,
							   qkv=self.qkv.new(), qkv_splits=self.qkv_splits,
							   proj=self.proj.new(), attn_drop=self.attn_drop.p, proj_drop=self.proj_drop.p)

	def aggregate(self, qkv):
		k, v = qkv
		return self.aggr(k, v)

	def attend(self, qkv):
		k, v = qkv
		k, v = self.tokenizer.untokenize(merge_heads(k)), self.tokenizer.untokenize(merge_heads(v))
		kernel_size = [v.shape[i+2] if s < 0 else s for i, s in enumerate(self.kernel_size)]
		stride = [v.shape[i+2] if s < 0 else s for i, s in enumerate(self.stride)]
		k = unfold(k, kernel_size=kernel_size, stride=stride, padding=self.padding)
		v = unfold(v, kernel_size=kernel_size, stride=stride, padding=self.padding)
		k, v = split_heads(self.tokenizer.tokenize(k), self.channels * utils.shape2size(kernel_size)), split_heads(self.tokenizer.tokenize(v), self.channels * utils.shape2size(kernel_size))
		k, v = (s.transpose(1, 2).reshape(s.shape[0], s.shape[2], s.shape[1], self.channels, -1).transpose(-2, -1) for s in (k, v))

		x = super(LocalAttention, self).attend((k, v))
		x = x.transpose(1, 2).transpose(-2, -1).reshape(x.shape[0], x.shape[2], x.shape[1], -1)

		if self.transposed and self.refold:
			x = self.tokenizer.untokenize(merge_heads(x))
			x = fold(x, kernel_size=kernel_size, stride=stride)
			x = split_heads(self.tokenizer.tokenize(x), self.channels)
		return x


class AttentionConv3dBlock(nn.Module):
	def __init__(self, channels=64, heads=8, attn=None, norm=BatchNorm, conv=None):
		super().__init__()
		self.channels = channels
		self.heads = heads
		self.attn = attn if attn is not None else Attention(self.channels, self.heads)
		self.conv = conv if conv is not None else MultiHeadConv3d(self.channels, self.channels, self.heads)
		self.norm = norm(self.heads * self.channels) if norm is not None else None

	def new(self):
		return AttentionConv3dBlock(self.channels, self.heads, attn=self.attn.new(), norm=self.norm.__class__, conv=self.conv.new())

	def forward(self, x):
		x = self.attn(x)
		x = self.conv(x if self.norm is None else self.norm(x))
		return x

class MultiHeadMLP(MultiHeadLinear):
	def __init__(self, in_features=512, hidden_features=512, out_features=512, heads=1, headsize=512, shared_map=False, bias=True,
				 hidden_layers=1, recurrent=False, act=nn.ReLU(), norm=BatchNorm, norm_before_act=False, token_dim=None, transposed=False, drop=0.):
		self.hidden_features = hidden_features
		self.hidden_layers = hidden_layers
		self.recurrent = recurrent
		self.layer_list = None
		self.act = act
		self.norm = norm
		self.norm_before_act = norm_before_act
		self.norm_layers = None
		self.drop = nn.Dropout(drop)
		super().__init__(in_features, out_features, heads, headsize, shared_map=shared_map,
						 bias=bias, token_dim=token_dim, transposed=transposed)

	def init_filters(self):
		self.layer_list = nn.ModuleList()
		self.norm_layers = nn.ModuleList()
		for l in range(self.hidden_layers + 1):
			in_features = self.in_features if l == 0 else self.hidden_features
			out_features = self.out_features if l == self.hidden_layers else self.hidden_features
			headsize = self.headsize if l == 0 else (self.hidden_features * self.headsize) // self.in_features
			block = block if self.recurrent and l > 1 and l < self.hidden_layers else MultiHeadLinear(
				in_features, out_features, self.heads, headsize, shared_map=self.shared_map,
				bias=self.use_bias, token_dim=self.token_dim, transposed=self.transposed)
			self.layer_list.append(block)
			norm = norm if self.recurrent and l > 1 and l < self.hidden_layers else self.norm(self.heads * headsize)
			if l < self.hidden_layers: self.norm_layers.append(norm)

	def new(self):
		return MultiHeadMLP(self.in_features, self.hidden_features, self.out_features, self.heads, self.headsize, shared_map=self.shared_map,
							bias=self.use_bias, hidden_layers=self.hidden_layers, recurrent=self.recurrent,
							act=self.act, norm=self.norm, norm_before_act=self.norm_before_act,
							token_dim=self.token_dim, transposed=self.transposed, drop=self.drop.p)

	def forward(self, x):
		for l in range(len(self.layer_list)):
			x = self.layer_list[l](x)
			if l < self.hidden_layers: x = self.drop(self.act(self.norm_layers[l](x)) if self.norm_before_act else self.norm_layers[l](self.act(x)))
		return x

class MultiHeadResBlock(nn.Module):
	def __init__(self, channels=512, heads=1, conv=None, act=nn.ReLU(), norm=BatchNorm, order='CNACNSA', drop=0.):
		super().__init__()
		self.channels = channels
		self.heads = heads
		self.conv = conv if conv is not None else MultiHeadConv3d(self.channels, self.channels, self.heads, padding='same')
		self.act = act
		self.norm = norm
		self.order = order.upper()
		for op in self.order:
			if op not in 'BCNAS': raise ValueError("Invalid operation order string: {}".format(order))
		self.drop = nn.Dropout(drop)

		self.init_filters()

	def init_filters(self):
		self.conv1 = self.conv
		self.norm1 = self.norm(self.heads * self.channels)
		self.conv2 = self.conv.new()
		self.norm2 = self.norm(self.heads * self.channels)

	def new(self):
		return MultiHeadResBlock(self.channels, self.heads, conv=self.conv.new(), act=self.act, norm=self.norm, order=self.order, drop=self.drop.p)

	def forward(self, x):
		res = x.clone()
		c_count = 0
		n_count = 0
		for op in self.order:
			if op == 'B':
				x = res
				res = x.clone()
			if op == 'C':
				c_count += 1
				res = getattr(self, 'conv' + str(c_count))(res)
			if op == 'N':
				n_count += 1
				res = getattr(self, 'norm' + str(n_count))(res)
			if op == 'A':
				res = self.act(res)
			if op == 'S':
				res = res + x
			if op == 'D':
				res = self.drop(res)
		return res


class CBlock(nn.Module):
	def __init__(self, in_channels=512, out_channels=512, heads=1,
				 conv=None, proj=None, act=nn.ReLU(), norm=BatchNorm, order='NCNAP', splits=None):
		super().__init__()
		self.heads = heads
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.splits = splits if splits is not None else [self.heads * self.out_channels]
		if sum(self.splits) != self.out_channels * self.heads:
			raise ValueError("Channel splits size ({}) does not match the total output channel size ({})".format(sum(self.splits), self.out_channels * self.heads))
		self.conv = conv if conv is not None else MultiHeadConv3d(self.in_channels * self.heads, sum(self.splits) // self.heads)
		self.proj = proj if proj is not None else Column(MultiHeadResBlock(self.out_channels, self.heads))
		self.act = act
		self.norm1 = norm((self.in_channels if any([order.upper().startswith(s) for s in ['N', 'AN']]) else self.out_channels) * self.heads)
		self.norm2 = norm(self.out_channels * self.heads)
		self.order = order.upper()
		for op in self.order:
			if op not in 'BCNAPS': raise ValueError("Invalid operation order string: {}".format(order))		

	def new(self):
		return CBlock(self.in_channels, self.out_channels, self.heads, conv=self.conv.new(), proj=self.proj.new(),
					  act=self.act, norm=self.norm1.__class__, order=self.order, splits=self.splits)

	def forward(self, x):
		res = x.clone()
		n_count = 0
		for op in self.order:
			if op == 'B':
				x = res
				res = x.clone()
			if op == 'C':
				res = self.conv(res)
			if op == 'N':
				n_count += 1
				res = getattr(self, 'norm' + str(n_count))(res)
			if op == 'A':
				res = self.act(res)
			if op == 'P':
				res = self.proj(res)
			if op == 'S':
				res = torch.cat([
					s.reshape(s.shape[0], -1, self.in_channels, *s.shape[2:]) +
					x.reshape(x.shape[0], -1, self.in_channels, *x.shape[2:])
						for s in channel_split(res, self.splits)
					], dim=2).reshape(x.shape[0], -1, *x.shape[2:])
		return res

class ABlock(nn.Module):
	def __init__(self, channels=64, heads=8, attn=None, proj=None, norm=BatchNorm, order='NASNBPS'):
		super().__init__()
		self.channels = channels
		self.heads = heads
		self.attn = attn if attn is not None else Attention(self.channels, self.heads)
		self.proj = proj if proj is not None else MultiHeadMLP(self.channels, self.channels, self.channels, self.heads)
		self.norm = norm
		self.norm1 = self.norm(self.channels * self.heads)
		self.norm2 = self.norm(self.channels * self.heads)
		self.order = order.upper()
		for op in self.order:
			if op not in 'BNAPS': raise ValueError("Invalid operation order string: {}".format(order))

	def new(self):
		return ABlock(self.channels, self.heads, attn=self.attn.new(), proj=self.proj.new(), norm=self.norm, order=self.order)

	def forward(self, x):
		res = x.clone()
		n_count = 0
		for op in self.order:
			if op == 'B':
				x = res
				res = x.clone()
			if op == 'N':
				n_count += 1
				res = getattr(self, 'norm' + str(n_count))(res)
			if op == 'A':
				res = self.attn(res)
			if op == 'P':
				res = self.proj(res)
			if op == 'S':
				res = res + x
		return res

class Column(nn.Module):
	def __init__(self, block, depth=1, recurrent=False):
		super().__init__()
		self.block = block
		self.depth = depth
		self.recurrent = recurrent

		self.layer_list = nn.ModuleList()
		for l in range(self.depth):
			self.layer_list.append(self.block if l == 0 or self.recurrent else self.block.new())

	def new(self):
		return Column(self.block.new(), depth=self.depth, recurrent=self.recurrent)

	def forward(self, x):
		for l in self.layer_list:
			x = l(x)
		return x


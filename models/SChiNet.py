import torch
import torch.nn as nn

from .modutils import *
import utils



class TubeTokenizer(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x

class ChiStream(nn.Module):
	def __init__(self, pw1_shape=(4, 32*32, 64), pw2_shape=(16, 32*32, 64), heads=8, token_dim=None, x_dim=(3, 2)):
		super().__init__()

		self.pw1_shape = pw1_shape
		self.pw1_compr_shape = tuple(pw2_shape[i] if i + 2 == x_dim[0] else s for i, s in enumerate(pw1_shape))
		self.pw2_shape = pw2_shape
		self.pw2_compr_shape = tuple(pw1_shape[i] if i + 2 == x_dim[1] else s for i, s in enumerate(pw2_shape))
		self.heads = heads
		self.headsize1 = self.pw1_shape[-1]
		self.headsize2 = self.pw2_shape[-1]
		self.pw1_mixes_channels = (x_dim[0] == (2 + len(self.pw1_shape) - 1))
		self.pw2_mixes_channels = (x_dim[1] == (2 + len(self.pw2_shape) - 1))
		self.channels1 = self.pw1_shape[x_dim[1] - 2] if self.pw1_mixes_channels else self.pw1_shape[-1]
		self.channels2 = self.pw2_shape[x_dim[0] - 2] if self.pw2_mixes_channels else self.pw1_shape[-1]
		self.tokens1 = (self.pw1_shape[x_dim[0] - 2] + self.pw1_compr_shape[x_dim[0] - 2]) if self.pw2_mixes_channels else ((self.pw1_shape[-1] + self.pw1_compr_shape[-1]) if self.pw1_mixes_channels else self.pw1_shape[-1])
		self.tokens2 = (self.pw2_shape[x_dim[1] - 2] + self.pw2_compr_shape[x_dim[1] - 2]) if self.pw1_mixes_channels else ((self.pw2_shape[-1] + self.pw2_compr_shape[-1]) if self.pw2_mixes_channels else self.pw2_shape[-1])
		self.token_dim = token_dim
		self.x_dim = x_dim

		c1 = self.channels1 * self.heads
		c2 = self.channels2 * self.heads
		t1 = self.tokens1 * self.heads
		t2 = self.tokens2 * self.heads

		# Attention stage 1 QKV mappings
		self.Q1_1 = nn.Linear(c1, c1)
		self.K1_1 = nn.Linear(c1, c1)
		self.V1_1 = nn.Linear(c1, c1)
		self.Q2_1 = nn.Linear(c2, c2)
		self.K2_1 = nn.Linear(c2, c2)
		self.V2_1 = nn.Linear(c2, c2)

		# Attention stage 2 QKV mappings
		self.Q1_2 = nn.Linear(c1, c1)
		self.K1_2 = nn.Linear(c1, c1)
		self.V1_2 = nn.Linear(c1, c1)
		self.Q2_2 = nn.Linear(c2, c2)
		self.K2_2 = nn.Linear(c2, c2)
		self.V2_2 = nn.Linear(c2, c2)

		# Attention stage 3 QKV mappings
		self.Q1_3 = nn.Linear(t1, t1)
		self.K1_3 = nn.Linear(t1, t1)
		self.V1_3 = nn.Linear(t1, t1)
		self.Q2_3 = nn.Linear(t2, t2)
		self.K2_3 = nn.Linear(t2, t2)
		self.V2_3 = nn.Linear(t2, t2)

	def new(self):
		return ChiStream(pw1_shape=self.pw1_shape, pw2_shape=self.pw2_shape, heads=self.heads, token_dim=self.token_dim, x_dim=self.x_dim)

	def proj(self, M, x):
		return M(
			x.transpose(0, -1).reshape(x.shape[-1]*x.shape[1], *x.shape[2:-1], x.shape[0]).transpose(0, -1)
		).transpose(0, -1).reshape(x.shape[-1], x.shape[1], *x.shape[2:-1], x.shape[0]).transpose(0, -1)

	def qkv_aggregate(self, q, k, v, dim=-2):
		return qkv_aggregate(q.transpose(dim, -2), k.transpose(dim, -2), v.transpose(dim, -2)).transpose(dim, -2)

	def _v1(self, x1, p1, x2, p2):
		# x1 --> Nc
		# p1 --> nc
		# x2 --> nC
		# p2 --> nc

		# Prepare
		x1 = torch.cat([x1, p2], dim=-2)
		x2 = torch.cat([x2, p1], dim=-1)

		# Attend
		x1 = self.qkv_aggregate(x1, x1, x1)
		x2 = self.qkv_aggregate(x2.transpose(-2, -1), x2.transpose(-2, -1), x2.transpose(-2, -1)).transpose(-2, 1)
		x1_hat, p2_hat = x1[:, :, :-p2.shape[-2], :], x1[:, :, -p2.shape[-2]:, :]
		x2_hat, p1_hat = x2[:, :, :, :p1.shape[-1]], x2[:, :, :, p1.shape[-1]:]

		# Chiasm
		x1_hat = torch.cat([x1_hat, p1_hat], dim=-2)
		x2_hat = torch.cat([x2_hat, p2_hat], dim=-1)

		# Attend again
		y1 = self.qkv_aggregate(x1_hat, x1_hat, x1_hat)
		y2 = self.qkv_aggregate(x2_hat.transpose(-2, -1), x2_hat.transpose(-2, -1), x2_hat.transpose(-2, -1)).transpose(-2, 1)
		y1, q1 = y1[:, :, :-p1.shape[-2], :], y1[:, :, -p1.shape[-2]:, :]
		y2, q2 = y2[:, :, :, :p2.shape[-1]], y2[:, :, :, p2.shape[-1]:]

		return y1, q1, y2, q2

	def _v2(self, x1, p1, x2, p2):
		# x1 --> Nc
		# p1 --> nc
		# x2 --> nC
		# p2 --> nc

		# Prepare
		x1 = torch.cat([x1, p1], dim=-2)
		x2 = torch.cat([x2, p2], dim=-1)

		# Compress
		x1_hat = self.qkv_aggregate(p2, x1, x1)
		x2_hat = self.qkv_aggregate(p1.transpose(-2, -1), x2.transpose(-2, -1), x2.transpose(-2, -1)).transpose(-2, -1)

		# Cross-attend
		x12 = torch.cat([x1_hat, x2_hat], dim=-2)
		x12 = self.qkv_aggregate(x12, x12, x12)
		x1_hat, x2_hat = x12[:, :, :-x2_hat.shape[-2], :], x12[:, :, -x2_hat.shape[-2]:, :]
		x12 = torch.cat([x1_hat, x2_hat], dim=-1)
		x12 = self.qkv_aggregate(x12.transpose(-2, -1), x12.transpose(-2, -1), x12.transpose(-2, -1)).transpose(-2, -1)
		x1_hat, x2_hat = x12[:, :, :, :x2_hat.shape[-1]], x12[:, :, :, x2_hat.shape[-1]:]

		# Expand
		y1 = self.qkv_aggregate(x1, x1_hat, x1_hat)
		y2 = self.qkv_aggregate(x2.transpose(-2, -1), x2_hat.transpose(-2, -1), x2_hat.transpose(-2, -1)).transpose(-2, -1)
		q1, q2 = x1_hat, x2_hat

		return y1, q1, y2, q2

	def _v3(self, x1, p1, x2, p2):
		# x1 --> Nc
		# p1 --> nc
		# x2 --> nC
		# p2 --> nc

		# Prepare inputs
		x1 = torch.cat([x1, p1], dim=self.x_dim[0])
		x2 = torch.cat([x2, p2], dim=self.x_dim[1])

		# Attention stage 1: Compress along high-res pathway
		x1_hat = self.qkv_aggregate(self.proj(self.Q1_1, p1), self.proj(self.K1_1, x1), self.proj(self.V1_1, x1), dim=self.x_dim[0])
		x2_hat = self.qkv_aggregate(self.proj(self.Q2_1, p2.transpose(self.x_dim[0], self.x_dim[1])), self.proj(self.K2_1, x2.transpose(self.x_dim[0], self.x_dim[1])), self.proj(self.V2_1, x2.transpose(self.x_dim[0], self.x_dim[1])), dim=self.x_dim[0]).transpose(self.x_dim[0], self.x_dim[1])

		# Chiasm
		x12_hat, x21_hat = torch.cat([x1_hat, x2_hat], dim=self.x_dim[0]), torch.cat([x2_hat, x1_hat], dim=self.x_dim[1])

		# Optional: Cat with uncompressed tensors and compress along alternate high-res pathway
		#x12 = torch.cat([x1, x12_hat], dim=-2)
		#x21 = torch.cat([x2, x21_hat], dim=-1)
		#x12_hat = self.qkv_aggregate(p1, x12, x12)
		#x21_hat = self.qkv_aggregate(p2.transpose(-2, -1), x21.transpose(-2, -1), x21.transpose(-2, -1)).transpose(-2, -1)
		#x12_hat, x21_hat = torch.cat([x12_hat, x21_hat], dim=self.x_dim[0]), torch.cat([x21_hat, x12_hat], dim=self.x_dim[1])

		# Attention stage 2: Extend along alternate high-res pathway
		x12 = self.qkv_aggregate(self.proj(self.Q1_2, x1), self.proj(self.K1_2, x12_hat), self.proj(self.V1_2, x12_hat), dim=self.x_dim[0])
		x21 = self.qkv_aggregate(self.proj(self.Q2_2, x2.transpose(self.x_dim[0], self.x_dim[1])), self.proj(self.K2_2, x21_hat.transpose(self.x_dim[0], self.x_dim[1])), self.proj(self.V2_2, x21_hat.transpose(self.x_dim[0], self.x_dim[1])), dim=self.x_dim[0]).transpose(self.x_dim[0], self.x_dim[1])

		# Cat along alternate low-res pathway
		x12 = torch.cat([x1, x12], dim=self.x_dim[1])
		x21 = torch.cat([x2, x21], dim=self.x_dim[0])

		# Attention stage 3: Aggregate along alternate low-res pathway
		y1 = self.qkv_aggregate(self.proj(self.Q1_3, x1.transpose(self.x_dim[0], self.x_dim[1])), self.proj(self.K1_3, x12.transpose(self.x_dim[0], self.x_dim[1])), self.proj(self.V1_3, x12.transpose(self.x_dim[0], self.x_dim[1])), dim=self.x_dim[0]).transpose(self.x_dim[0], self.x_dim[1])
		y1, q1 = torch.split(y1, (y1.shape[self.x_dim[0]] - p1.shape[self.x_dim[0]], p1.shape[self.x_dim[0]]), dim=self.x_dim[0]) #y1[:, :, :-p1.shape[-2], :], y1[:, :, -p1.shape[-2]:, :]
		y2 = self.qkv_aggregate(self.proj(self.Q2_3, x2), self.proj(self.K2_3, x21), self.proj(self.V2_3, x21), dim=self.x_dim[0])
		y2, q2 = torch.split(y2, (y2.shape[self.x_dim[1]] - p2.shape[self.x_dim[1]], p2.shape[self.x_dim[1]]), dim=self.x_dim[1]) #y2[:, :, :, :-p2.shape[-1]], y2[:, :, :, -p2.shape[-1]:]

		return y1, q1, y2, q2

	def _tokenize(self, x, token_dim, pw_shape):
		tokenizer = Tokenizer(list(range(x.ndim)), token_dim)
		x = split_heads(tokenizer.tokenize(x), pw_shape[-1])
		return x.reshape(x.shape[0], x.shape[1], *pw_shape), tokenizer, pw_shape

	def _untokenize(self, x, tokenizer, pw_shape):
		return tokenizer.untokenize(merge_heads(x.reshape(x.shape[0], x.shape[1], -1, pw_shape[-1])))

	def forward(self, x1, p1, x2, p2):
		x1, p1, x2, p2 = (self._tokenize(x, self.token_dim, pws) for x, pws in zip((x1, p1, x2, p2), (self.pw1_shape, self.pw1_compr_shape, self.pw2_shape, self.pw2_compr_shape)))
		return (self._untokenize(x, t, pws) for x, t, pws in zip(self._v3(x1[0], p1[0], x2[0], p2[0]), (x1[1], p1[1], x2[1], p2[1]), (x1[2], p1[2], x2[2], p2[2])))

class MixedResTubeletEnc(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		pass

class SChiNet(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x



def test_chistream():
	B, s, C, c, T, t, H, W, h, w = 3, 2, 64, 8, 16, 4, 24, 24, 6, 6
	x1 = torch.randn(B, s*c, T, H, W)
	p1 = torch.randn(B, s*c, t, H, W)
	x2 = torch.randn(B, s*C, t, H, W)
	p2 = torch.randn(B, s*c, t, H, W)

	print("Input shapes: x1 {}, p1 {}, x2 {}, p2 {}".format(x1.shape, p1.shape, x2.shape, p2.shape))

	chi = ChiStream(pw1_shape=(T, H*W, c), pw2_shape=(t, H*W, C), heads=s, token_dim=(2, 3, 4), x_dim=(2, 4))

	y1, q1, y2, q2 = chi(x1, p1, x2, p2)

	print("Output shapes: x1 {}, p1 {}, x2 {}, p2 {}".format(y1.shape, q1.shape, y2.shape, q2.shape))


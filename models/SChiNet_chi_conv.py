import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from .modutils import *
import utils


class TokenBatchNorm(nn.BatchNorm1d):
	def forward(self, x):
		return super().forward(x.reshape(-1, x.shape[-1])).reshape(*x.shape)

dim2xdim = {1: 4, 2: 2, 3: 3, 4: 3}
strxdim2xdim = {'t': 2, 's': 3, 'c': 4}

class ChiStream(nn.Module):
	def __init__(self, pw1_shape=(64, 8, 14, 14), pw2_shape=(64, 32, 7, 7), heads=8, token_dim=None, x_dim=('s', 't'),
	             norm=nn.LayerNorm, headwise_map=False, alternate_attn=False):
		super().__init__()

		self.debug = False

		x_dim = [strxdim2xdim[d] for d in x_dim]
		self.pw1_shape = pw1_shape
		self.pw1_tokenized_shape = [pw1_shape[1], pw1_shape[2]*pw1_shape[3], pw1_shape[0]]
		self.pw1_compr_shape = [pw2_shape[i] if dim2xdim[i+1] == x_dim[0] else s if dim2xdim[i+1] in x_dim or dim2xdim[i+1]==4 else 1 for i, s in enumerate(pw1_shape)]
		self.pw1_tokenized_compr_shape = [self.pw1_compr_shape[1], self.pw1_compr_shape[2]*self.pw1_compr_shape[3], self.pw1_compr_shape[0]]
		self.pw2_shape = pw2_shape
		self.pw2_tokenized_shape = [pw2_shape[1], pw2_shape[2] * pw2_shape[3], pw2_shape[0]]
		self.pw2_compr_shape = [pw1_shape[i] if dim2xdim[i+1] == x_dim[1] else s if dim2xdim[i+1] in x_dim or dim2xdim[i+1]==4 else 1 for i, s in enumerate(pw2_shape)]
		self.pw2_tokenized_compr_shape = [self.pw2_compr_shape[1], self.pw2_compr_shape[2]*self.pw2_compr_shape[3], self.pw2_compr_shape[0]]
		self.heads = heads
		self.norm = norm
		self.headwise_map = headwise_map
		self.alternate_attn = alternate_attn

		self.headsize1 = pw1_shape[0]
		self.headsize2 = pw2_shape[0]
		self.pw1_mixes_channels = (x_dim[0] == 4)
		self.pw2_mixes_channels = (x_dim[1] == 4)
		self.channels1 = pw1_shape[0]
		self.channels2 = pw2_shape[0]
		self.tokens1 = (self.pw1_tokenized_shape[x_dim[0] - 2] + self.pw1_tokenized_compr_shape[x_dim[0] - 2]) if self.pw2_mixes_channels else ((self.pw1_tokenized_shape[-1] + self.pw1_tokenized_compr_shape[-1]) if self.pw1_mixes_channels else self.headsize1)
		self.tokens2 = (self.pw2_tokenized_shape[x_dim[1] - 2] + self.pw2_tokenized_compr_shape[x_dim[1] - 2]) if self.pw1_mixes_channels else ((self.pw2_tokenized_shape[-1] + self.pw2_tokenized_compr_shape[-1]) if self.pw2_mixes_channels else self.headsize2)
		self.token_dim = token_dim
		self.x_dim = x_dim

		c1 = self.headsize1 * self.heads
		c2 = self.headsize2 * self.heads
		t1 = self.tokens1 * self.heads
		t2 = self.tokens2 * self.heads

		# Convolutional mappings
		d1, h1, w1 = pw1_shape[1], pw1_shape[2], pw1_shape[3]
		d2, h2, w2 = pw2_shape[1], pw2_shape[2], pw2_shape[3]
		c_inner = 2 * min(c1, c2)

		self.compress1 = nn.Sequential(
				nn.Conv3d(c1, c_inner, 3, padding=1),
				nn.Upsample((d2, h2, w2)),
				nn.ReLU(inplace=True),
				nn.BatchNorm3d(c_inner),
			)

		self.compress2 = nn.Sequential(
				nn.Conv3d(c2, c_inner, 3, padding=1),
				nn.Upsample((d1, h1, w1)),
				nn.ReLU(inplace=True),
				nn.BatchNorm3d(c_inner),
			)

		self.compress_b1 = nn.Sequential(
				nn.Conv3d(c1, c_inner, 3, padding=1),
				nn.Upsample((min(d1, d2) if 2 in self.x_dim else 1, min(h1, h2) if 3 in self.x_dim else 1, min(w1, w2) if 3 in self.x_dim else 1)),
				nn.ReLU(inplace=True),
				nn.BatchNorm3d(c_inner),
			)

		self.compress_b2 = nn.Sequential(
				nn.Conv3d(c2, c_inner, 3, padding=1),
				nn.Upsample((min(d1, d2) if 2 in self.x_dim else 1, min(h1, h2) if 3 in self.x_dim else 1, min(w1, w2) if 3 in self.x_dim else 1)),
				nn.ReLU(inplace=True),
				nn.BatchNorm3d(c_inner),
			)

		#self.out1 = nn.Sequential(
		#		nn.Conv3d(c1 + c_inner, c1, 1),
		#		nn.ReLU(inplace=True),
		#		nn.BatchNorm3d(c1),
		#	)

		#self.out2 = nn.Sequential(
		#		nn.Conv3d(c2 + c_inner, c2, 1),
		#		nn.ReLU(inplace=True),
		#		nn.BatchNorm3d(c2),
		#	)

		# self.out_b1 = nn.Sequential(
		#		nn.Conv3d(c1 + c_inner, c1, 1),
		#		nn.ReLU(inplace=True),
		#		nn.BatchNorm3d(c1),
		#	)

		# self.out_b2 = nn.Sequential(
		#		nn.Conv3d(c2 + c_inner, c2, 1),
		#		nn.ReLU(inplace=True),
		#		nn.BatchNorm3d(c2),
		#	)

	def new(self):
		return ChiStream(pw1_shape=self.pw1_shape, pw2_shape=self.pw2_shape, heads=self.heads, token_dim=self.token_dim, x_dim=self.x_dim,
		                 norm=self.norm, headwise_map=self.headwise_map, alternate_attn=self.alternate_attn)

	def chi(self, x1, p1, x2, p2):
		x12, x21 = self.compress1(x1), self.compress2(x2)
		p12, p21 = self.compress_b1(x1), self.compress_b2(x2)
		y1, y2 = torch.cat([x1, x21], dim=1), torch.cat([x2, x12], dim=1)
		q1, q2 = torch.cat([p1, p21], dim=1), torch.cat([p2, p12], dim=1)
		#y1, y2 = self.out1(y1) + x1, self.out2(y2) + x2
		return y1, q1, y2, q2

	def forward(self, x1, p1, x2, p2):
		return self.chi(x1, p1, x2, p2)

	def set_debug_mode(self, mode=False):
		self.debug = mode

class MixedResTubeletEnc(nn.Module):
	def __init__(self, in_channels, out_channels1, kernel_size1, stride1, out_channels2, kernel_size2, stride2, res_kernel_size, act, norm, x_dim=('s', 't')):
		super().__init__()
		self.x_dim = x_dim

		self.emb1 = nn.Sequential(
				CBlock(in_channels, out_channels1, 1,
					conv=MultiHeadConv3d(in_channels, out_channels1, 1, in_channels,
					       shared_map=False, kernel_size=kernel_size1, stride=stride1, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels1, 1,
						conv=MultiHeadConv3d(out_channels1, out_channels1, 1, out_channels1, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(in_channels, out_channels1, kernel_size1, stride1, utils.get_padding_same(kernel_size1)),
				#nn.ReLU(),
				#nn.BatchNorm3d(out_channels1),
				CBlock(out_channels1, 2*out_channels1, 1,
					conv=MultiHeadConv3d(out_channels1, 2*out_channels1, 1, out_channels1,
					       shared_map=False, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='same'),
					proj=Column(MultiHeadResBlock(2*out_channels1, 1,
						conv=MultiHeadConv3d(2*out_channels1, 2*out_channels1, 1, 2*out_channels1, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(out_channels1, 2*out_channels1, 3, 2, utils.get_padding_same(3)),
				#nn.ReLU(),
				#nn.BatchNorm3d(2*out_channels1),
			)
		self.pos_embed1 = nn.Parameter(torch.empty([]), requires_grad=True)
		self.p1 = nn.Parameter(torch.empty([]), requires_grad=True)

		kernel_size1_p = (max(k1, k2) for k1, k2 in zip(kernel_size1, kernel_size2))
		stride1_p = (max(s1, s2) for s1, s2 in zip(stride1, stride2))
		self.emb1_p = nn.Sequential(
				CBlock(in_channels, out_channels1, 1,
					conv=MultiHeadConv3d(in_channels, out_channels1, 1, in_channels,
					       shared_map=False, kernel_size=kernel_size1_p, stride=stride1_p, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels1, 1,
						conv=MultiHeadConv3d(out_channels1, out_channels1, 1, out_channels1, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(in_channels, out_channels1, kernel_size1, stride1, utils.get_padding_same(kernel_size1)),
				#nn.ReLU(),
				#nn.BatchNorm3d(out_channels1),
				CBlock(out_channels1, 2*out_channels1, 1,
					conv=MultiHeadConv3d(out_channels1, 2*out_channels1, 1, out_channels1,
					       shared_map=False, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='same'),
					proj=Column(MultiHeadResBlock(2*out_channels1, 1,
						conv=MultiHeadConv3d(2*out_channels1, 2*out_channels1, 1, 2*out_channels1, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(out_channels1, 2*out_channels1, 3, 2, utils.get_padding_same(3)),
				#nn.ReLU(),
				#nn.BatchNorm3d(2*out_channels1),
			)

		self.emb2 = nn.Sequential(
				CBlock(in_channels, out_channels2, 1,
					conv=MultiHeadConv3d(in_channels, out_channels2, 1, in_channels,
					       shared_map=False, kernel_size=kernel_size2, stride=stride2, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels2, 1,
						conv=MultiHeadConv3d(out_channels2, out_channels2, 1, out_channels2, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(in_channels, out_channels2, kernel_size2, stride2, utils.get_padding_same(kernel_size2)),
				#nn.ReLU(),
				#nn.BatchNorm3d(out_channels2),
				CBlock(out_channels2, 2*out_channels2, 1,
					conv=MultiHeadConv3d(out_channels2, 2*out_channels2, 1, out_channels2,
					       shared_map=False, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='same'),
					proj=Column(MultiHeadResBlock(2*out_channels2, 1,
						conv=MultiHeadConv3d(2*out_channels2, 2*out_channels2, 1, 2*out_channels2, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(out_channels2, 2*out_channels2, 3, 2, utils.get_padding_same(3)),
				#nn.ReLU(),
				#nn.BatchNorm3d(2*out_channels2),
			)
		self.pos_embed2 = nn.Parameter(torch.empty([]), requires_grad=True)
		self.p2 = nn.Parameter(torch.empty([]), requires_grad=True)

		kernel_size2_p = (max(k1, k2) for k1, k2 in zip(kernel_size1, kernel_size2))
		stride2_p = (max(s1, s2) for s1, s2 in zip(stride1, stride2))
		self.emb2_p = nn.Sequential(
				CBlock(in_channels, out_channels1, 1,
					conv=MultiHeadConv3d(in_channels, out_channels1, 1, in_channels,
					       shared_map=False, kernel_size=kernel_size2_p, stride=stride2_p, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels1, 1,
						conv=MultiHeadConv3d(out_channels1, out_channels1, 1, out_channels1, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(in_channels, out_channels1, kernel_size1, stride1, utils.get_padding_same(kernel_size1)),
				#nn.ReLU(),
				#nn.BatchNorm3d(out_channels1),
				CBlock(out_channels1, 2*out_channels1, 1,
					conv=MultiHeadConv3d(out_channels1, 2*out_channels1, 1, out_channels1,
					       shared_map=False, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='same'),
					proj=Column(MultiHeadResBlock(2*out_channels1, 1,
						conv=MultiHeadConv3d(2*out_channels1, 2*out_channels1, 1, 2*out_channels1, shared_map=False,
							   kernel_size=(1, res_kernel_size, res_kernel_size), stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=0, recurrent=False),
		        	act=act, norm=norm, order='CANPN'),
				#nn.Conv3d(out_channels1, 2*out_channels1, 3, 2, utils.get_padding_same(3)),
				#nn.ReLU(),
				#nn.BatchNorm3d(2*out_channels1),
			)

	def forward(self, x):
		p_pool_dims = [ i for i in range(x.ndim) if i > 1 and dim2xdim[i] not in [strxdim2xdim[d] for d in self.x_dim] ]
		x1, p1, x2, p2 = self.emb1(x), self.emb1_p(x), self.emb2(x), self.emb2_p(x)
		if len(p_pool_dims) > 0: p1, p2 = p1.mean(dim=p_pool_dims, keepdims=True), p2.mean(dim=p_pool_dims, keepdims=True)
		p_shape = [ 1 if i > 0 and dim2xdim[i+1] not in [strxdim2xdim[d] for d in self.x_dim] else min(s1, s2) for i, (s1, s2) in enumerate(zip(x1.shape[1:], x2.shape[1:])) ]
		if self.pos_embed1.ndim == 0:
			self.pos_embed1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(*x1.shape[1:])), requires_grad=True)
		if self.p1.ndim == 0:
			self.p1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(p_shape)), requires_grad=True)
		if self.pos_embed2.ndim == 0:
			self.pos_embed2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(*x2.shape[1:])), requires_grad=True)
		if self.p2.ndim == 0:
			self.p2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(p_shape)), requires_grad=True)
		return self.pos_embed1.unsqueeze(0) + x1, self.p1.unsqueeze(0) + p1, self.pos_embed2.unsqueeze(0) + x2, self.p2.unsqueeze(0) + p2

class SChiStage(nn.Module):
	def __init__(self, in_channels1, out_channels1, in_channels2, out_channels2, heads=1,
	             kernel_size=3, stride1=1, stride2=1, res_kernel_size=1, depth=2, fullconv=False, shared_map=False, headwise_map=False,
	             token_dim=None, x_dim=('s', 't'), alternate_attn=False, act=nn.ReLU, norm=BatchNorm, chi_norm=nn.LayerNorm):
		super().__init__()

		self.debug = False

		self.in_channels1 = in_channels1
		self.in_channels2 = in_channels2
		self.out_channels1 = out_channels1
		self.out_channels2 = out_channels2
		self.heads = heads
		self.kernel_size = kernel_size
		self.stride1 = stride1
		self.stride2 = stride2
		self.res_kernel_size = res_kernel_size
		self.depth = depth
		self.fullconv = fullconv
		self.shared_map = shared_map
		self.headwise_map = headwise_map
		self.token_dim = token_dim
		self.x_dim = x_dim
		self.alternate_attn = alternate_attn
		self.act = act
		self.norm = norm
		self.chi_norm = chi_norm

		self.pw1_shape = None
		self.pw2_shape = None
		self.chi = None

		in_channels_b, out_channels_b = min(self.in_channels1, self.in_channels2), min(self.out_channels1, self.out_channels2)
		sp = tuple((min(s1, s2) for s1, s2 in zip(_triple(stride1), _triple(stride2))))
		kernel_size_b = tuple(1 if dim2xdim[i+2] not in self.x_dim else k for i, k in enumerate(_triple(kernel_size)))
		res_kernel_size_b = tuple(1 if dim2xdim[i+2] not in self.x_dim else k for i, k in enumerate(_triple(res_kernel_size)))
		inner_channels1 = in_channels1 + ((2 * in_channels_b) if self.x_dim is not None else 0)
		inner_channels2 = in_channels2 + ((2 * in_channels_b) if self.x_dim is not None else 0)
		inner_channels_b = in_channels_b + ((2 * in_channels_b) if self.x_dim is not None else 0)

		self.pw1_a = CBlock(inner_channels1, out_channels1, heads,
					conv=MultiHeadConv3d(*((inner_channels1*heads, out_channels1*heads, 1, inner_channels1*heads) if fullconv else (inner_channels1, out_channels1, heads, inner_channels1)),
					       shared_map=shared_map, kernel_size=kernel_size, stride=stride1, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels1, heads,
						conv=MultiHeadConv3d(out_channels1, out_channels1, heads, out_channels1, shared_map=shared_map,
							   kernel_size=res_kernel_size, stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=depth, recurrent=False),
		            act=act, norm=norm, order='CANPN')

		self.pw1_b = CBlock(inner_channels_b, out_channels_b, heads,
					conv=MultiHeadConv3d(*((inner_channels_b*heads, out_channels_b*heads, 1, inner_channels_b*heads) if fullconv else (inner_channels_b, out_channels_b, heads, inner_channels_b)),
					       shared_map=shared_map, kernel_size=kernel_size_b, stride=sp, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels_b, heads,
						conv=MultiHeadConv3d(out_channels_b, out_channels_b, heads, out_channels_b, shared_map=shared_map,
							   kernel_size=res_kernel_size_b, stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=depth, recurrent=False),
		            act=act, norm=norm, order='CANPN')

		self.pw2_a = CBlock(inner_channels2, out_channels2, heads,
					conv=MultiHeadConv3d(*((inner_channels2*heads, out_channels2*heads, 1, inner_channels2*heads) if fullconv else (inner_channels2, out_channels2, heads, inner_channels2)),
					       shared_map=shared_map, kernel_size=kernel_size, stride=stride2, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels2, heads,
						conv=MultiHeadConv3d(out_channels2, out_channels2, heads, out_channels2, shared_map=shared_map,
							   kernel_size=res_kernel_size, stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=depth, recurrent=False),
		            act=act, norm=norm, order='CANPN')

		self.pw2_b = CBlock(in_channels_b, out_channels_b, heads,
					conv=MultiHeadConv3d(*((inner_channels_b*heads, out_channels_b*heads, 1, inner_channels_b*heads) if fullconv else (inner_channels_b, out_channels_b, heads, inner_channels_b)),
					       shared_map=shared_map, kernel_size=kernel_size_b, stride=sp, padding='same'),
					proj=Column(MultiHeadResBlock(out_channels_b, heads,
						conv=MultiHeadConv3d(out_channels_b, out_channels_b, heads, out_channels_b, shared_map=shared_map,
							   kernel_size=res_kernel_size_b, stride=1, padding='same'),
						act=act, norm=norm, order='CANCANS'), depth=depth, recurrent=False),
		            act=act, norm=norm, order='CANPN')
		
		# Disable chi block
		#self.x_dim = None

	def forward(self, x1, p1, x2, p2):
		if self.chi is None and self.x_dim is not None:
			self.pw1_shape = (x1.shape[1] // self.heads, x1.shape[2], x1.shape[3], x1.shape[4])
			self.pw2_shape = (x2.shape[1] // self.heads, x2.shape[2], x2.shape[3], x2.shape[4])
			self.chi = ChiStream(self.pw1_shape, self.pw2_shape, self.heads, self.token_dim, self.x_dim, self.chi_norm, headwise_map=self.headwise_map, alternate_attn=self.alternate_attn)
			if not self.training: self.chi.eval()

		y1, q1, y2, q2 = self.chi(x1, p1, x2, p2) if self.x_dim is not None else (x1, p1, x2, p2)
		return self.pw1_a(y1), self.pw1_b(q1), self.pw2_a(y2), self.pw2_b(q2)

	def set_debug_mode(self, mode=False):
		self.debug = mode
		if self.chi is not None: self.chi.set_debug_mode(mode)

class SChiNet(nn.Module):
	def __init__(self, config, num_classes):
		super().__init__()

		self.debug = False

		self.clip_len = config.get('clip_len', 32)
		self.img_size = config.get('input_size', 112)
		self.fmap_size = config.get('fmap_size', (1, 1))
		self.num_classes = num_classes
		self.disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', False)
		enc_k1, enc_k2 = config.get('enc_kernel_sizes', ((4, 7, 7), (3, 7, 7)))
		enc_s1, enc_s2 = config.get('enc_strides', ((4, 2, 2), (2, 4, 4)))
		enc_c1, enc_c2 = config.get('enc_channels', (64, 64))
		stages = config.get('chi_stages', ((64, 64, 4, 2), (64, 64, 8, 2)))
		token_dim = config.get('token_dim', (2, 3, 4))
		x_dim = config.get('x_dim', ('s', 't'))
		shrink_min_dim = config.get('shrink_min_dim', True)
		alternate_attn = config.get('alternate_attn', False)
		fullconv = config.get('fullconv', False)
		shared_map = config.get('shared_map', False)
		headwise_map = config.get('headwise_map', False)
		head_regroup_post = config.get('head_regroup_post', False)
		r_k = config.get('res_kernel_size', 3)
		act = utils.retrieve(config.get('act', 'torch.nn.ReLU'))()
		norm = utils.retrieve(config.get('norm', 'models.modutils.BatchNorm'))
		chi_norm = utils.retrieve(config.get('chi_norm', 'torch.nn.LayerNorm'))
		init_mode = config.get('init_mode', 'kaiming_normal')
		final_drop = config.get('final_drop', 0.)

		self.head_regroup_post = head_regroup_post
		self.stages = stages
		heads = 1
		channels1 = enc_c1
		channels2 = enc_c2
		self.enc = MixedResTubeletEnc(3, channels1, enc_k1, enc_s1, channels2, enc_k2, enc_s2, r_k, act, norm, x_dim)

		img_size1 = utils.get_conv_output_size(self.img_size, enc_k1[1], enc_s1[1], padding='same')
		img_size1 = utils.get_conv_output_size(img_size1, 3, 2, padding='same')
		clip_len1 = utils.get_conv_output_size(self.clip_len, enc_k1[0], enc_s1[0], padding='same')
		clip_len1 = utils.get_conv_output_size(clip_len1, 1, 1, padding='same')
		img_size2 = utils.get_conv_output_size(self.img_size, enc_k2[1], enc_s2[1], padding='same')
		img_size2 = utils.get_conv_output_size(img_size2, 3, 2, padding='same')
		clip_len2 = utils.get_conv_output_size(self.clip_len, enc_k2[0], enc_s2[0], padding='same')
		clip_len2 = utils.get_conv_output_size(clip_len2, 1, 1, padding='same')
		channels1 = 2 * channels1
		channels2 = 2 * channels2

		layers = []
		x_start = 0
		for i, stage in enumerate(stages):
			out_embed_dim1, out_embed_dim2, out_heads, depth = stage
			in_heads = heads
			if not head_regroup_post: heads = out_heads
			channels1 = (channels1 * in_heads) // heads
			out_channels1 = (out_embed_dim1 * out_heads) // heads
			channels2 = (channels2 * in_heads) // heads
			out_channels2 = (out_embed_dim2 * out_heads) // heads

			if shrink_min_dim:
				s1 = (2 if min(clip_len1, clip_len2) > self.fmap_size[0] else 1, 2 if min(img_size1, img_size2) > self.fmap_size[1] else 1, 2 if min(img_size1, img_size2) > self.fmap_size[1] else 1)
				s2 = s1
			else:
				s1 = (2 if clip_len1 > self.fmap_size[0] else 1, 2 if img_size1 > self.fmap_size[1] else 1, 2 if img_size1 > self.fmap_size[1] else 1)
				s2 = (2 if clip_len2 > self.fmap_size[0] else 1, 2 if img_size2 > self.fmap_size[1] else 1, 2 if img_size2 > self.fmap_size[1] else 1)

			layers.append(SChiStage(channels1, out_channels1, channels2, out_channels2, heads,
                    kernel_size=3, stride1=s1, stride2=s2, res_kernel_size=r_k, depth=depth, fullconv=fullconv, shared_map=shared_map, headwise_map=headwise_map,
                    token_dim=token_dim, x_dim=(x_dim if i >= x_start else None), alternate_attn=alternate_attn, act=act, norm=norm, chi_norm=chi_norm)
            )

			img_size1 = utils.get_conv_output_size(img_size1, 3, s1[1], padding='same')
			clip_len1 = utils.get_conv_output_size(clip_len1, 3, s1[0], padding='same')
			img_size2 = utils.get_conv_output_size(img_size2, 3, s2[1], padding='same')
			clip_len2 = utils.get_conv_output_size(clip_len2, 3, s2[0], padding='same')

			channels1 = out_embed_dim1
			channels2 = out_embed_dim2
			heads = out_heads

		self.layers = nn.ModuleList(layers)

		self.pool = nn.AdaptiveAvgPool3d(1)

		fmap_shape = utils.get_fmap_shape(self)
		self.final_norm = norm(fmap_shape[0])
		self.final_drop = nn.Dropout(final_drop)
		self.clf = nn.Linear(utils.shape2size(fmap_shape), self.num_classes)

		utils.init_model_params(self, mode=init_mode)

	def get_default_input_shape(self):
		return (3, self.clip_len, self.img_size, self.img_size)

	def forward_features(self, x):
		x1, p1, x2, p2 = self.enc(x)
		#heads = 1
		#channels1, channels2 = x1.shape[1], x2.shape[1]
		if self.debug: print("Enc: x1 {}, p1 {}, x2 {}, p2 {}".format(x1.shape, p1.shape, x2.shape, p2.shape))
		for i, l in enumerate(self.layers):
			#out_embed_dim1, out_embed_dim2, out_heads, depth = self.stages[i]
			#in_heads = heads
			#if not self.head_regroup_post: heads = out_heads
			#in_channels1 = channels1
			#channels1 = (channels1 * in_heads) // heads
			#out_channels1 = (out_embed_dim1 * out_heads) // heads
			#in_channels2 = channels2
			#channels2 = (channels2 * in_heads) // heads
			#out_channels2 = (out_embed_dim2 * out_heads) // heads

			#if not self.head_regroup_post:
			#	x1, p1, x2, p2 = regroup_heads(x1, in_channels1, channels1), regroup_heads(p1, min(in_channels1, in_channels2), min(channels1, channels2)), regroup_heads(x2, in_channels2, channels2), regroup_heads(p2, min(in_channels1, in_channels2), min(channels1, channels2))

			x1, p1, x2, p2 = l(x1, p1, x2, p2)

			#channels1 = out_embed_dim1
			#channels2 = out_embed_dim2
			#heads = out_heads

			#if self.head_regroup_post:
			#	x1, p1, x2, p2 = regroup_heads(x1, out_channels1, channels1),  regroup_heads(p1, min(out_channels1, out_channels2), min(channels1, channels2)), regroup_heads(x2, in_channels2, channels2), regroup_heads(p2, min(out_channels1, out_channels2), min(channels1, channels2))

			if self.debug: print("Stage {}: x1 {}, p1 {}, x2 {}, p2 {}".format(i, x1.shape, p1.shape, x2.shape, p2.shape))

		return torch.cat([self.pool(x1), self.pool(p1), self.pool(x2), self.pool(p2)], dim=1)

	def forward(self, x):
		x = self.forward_features(x)
		x = self.final_drop(self.final_norm(x))
		return self.clf(x.reshape(x.shape[0], -1))

	def set_debug_mode(self, mode=False):
		self.debug = mode
		for l in self.layers: l.set_debug_mode(mode)




def test_schinet():
	channels1, channels2 = 64, 64
	defaults = {'clip_len': 32, 'input_size': 56, 'chi_stages': ((channels1, channels2, 4, 2), (channels1, channels2, 4, 2), (channels1, channels2, 8, 2), (channels1, channels2, 8, 2)), 'fmap_size': (3, 3), 'head_regroup_post': False, 'chi_norm': 'models.SChiNet.TokenBatchNorm',}
	#defaults = {'clip_len': 32, 'input_size': 56, 'chi_stages': ((channels1, channels2, 2, 2), (channels1, channels2, 4, 2), (channels1, channels2, 8, 2)), 'fmap_size': (3, 3), 'head_regroup_post': True}
	config1 = {**defaults, 'enc_kernel_sizes': ((8, 7, 7), (3, 7, 7)), 'enc_strides': ((8, 2, 2), (2, 4, 4)), 'enc_channels': (channels1, channels2), 'token_dim': (2, 3, 4), 'x_dim': ('s', 't')}
	config2 = {**defaults, 'enc_kernel_sizes': ((3, 7, 7), (8, 7, 7)), 'enc_strides': ((2, 2, 2), (8, 2, 2)), 'enc_channels': (channels1, channels2), 'token_dim': (2, 3, 4), 'x_dim': ('t', 'c')}
	config3 = {**defaults, 'enc_kernel_sizes': ((3, 7, 7), (3, 7, 7)), 'enc_strides': ((2, 2, 2), (2, 4, 4)), 'enc_channels': (channels1, channels2), 'token_dim': (2, 3, 4), 'x_dim': ('s', 'c')}
	config4 = {**defaults, 'enc_kernel_sizes': ((8, 7, 7), (3, 7, 7)), 'enc_strides': ((8, 2, 2), (2, 4, 4)), 'enc_channels': (channels1, channels2), 'token_dim': (2, 3, 4), 'x_dim': None}


	config = config3
	model = SChiNet(config, 101)
	model.set_debug_mode(True)

	model.to('cuda:0')

	num_params = 0
	for n, p in model.named_parameters():
		print("{}: {} ({})".format(n, p.numel(), p.shape))
		num_params += p.numel()
	print("Total params: {:.2f}M".format(num_params/1000000))

	x = torch.randn(5, 3, defaults['clip_len'], defaults['input_size'], defaults['input_size'], device='cuda:0')

	y = model(x)

	print(y.shape)


import torch
import torch.nn as nn

from .modutils import *
import utils


class CA3D(nn.Module):
	def __init__(self, config, num_classes):
		super().__init__()

		clip_len = config.get('clip_len', 32)
		img_size = config.get('input_size', 112)
		fmap_size = config.get('fmap_size', (1, 1))
		patch_size = config.get('patch_size', 3)
		layer_sizes = config.get('layer_sizes', ((64, 4, 2), (64, 8, 2)))
		conv_order = config.get('conv_order', 'NCNAP')
		res_order = config.get('res_order', 'CNACNSA')
		att_order = config.get('att_order', 'NASNBPS')
		r_k = config.get('res_kernel_size', 3)
		rec_col = config.get('rec_col', False)
		shared_map = config.get('shared_map', False)
		cape_reg = config.get('cape_reg', 0)
		act = utils.retrieve(config.get('act', 'torch.nn.ReLU'))()
		norm = utils.retrieve(config.get('norm', 'models.modutils.BatchNorm'))
		downsample = config.get('downsample', (1, 2, 2, 2, 3))
		t_aggr = config.get('t_aggr', 'conv')
		t_att = config.get('t_att', 'once')
		init_mode = config.get('init_mode', 'kaiming_normal')
		pos_emb_mode = config.get('pos_emb_mode', 'repeat')
		pos_drop = config.get('pos_drop', 0.)
		final_drop = config.get('final_drop', 0.)
		disable_wd_for_pos_emb = config.get('disable_wd_for_pos_emb', False)

		self.disable_wd_for_pos_emb = disable_wd_for_pos_emb
		self.clip_len = clip_len
		self.img_size = img_size
		self.fmap_size = fmap_size
		self.num_classes = num_classes
		self.layer_list = nn.ModuleList()

		# First embedding layer: spatio-temporal convolutional layer with residual column
		s_t, s = downsample[0], downsample[1]
		k_t = max(downsample[4], s_t)
		self.layer_list.append(CBlock(3, 64, 1,
			conv=make_conv_layer(config, 3, 64, 1, 3, kernel_size=(k_t, 7, 7), stride=(s_t, s, s), padding='same'),
			proj=Column(MultiHeadResBlock(64, 1,
				conv=make_conv_layer(config, 64, 64, 1, 64, kernel_size=(r_k if k_t > 1 else 1, r_k, r_k), stride=1, padding='same'),
				act=act, norm=norm, order=res_order), depth=2, recurrent=rec_col),
            act=act, norm=norm, order=conv_order))
		clip_len = clip_len // s_t
		img_size = img_size // s

		# Second layer: spatio-temporal convolutional layer with residual column
		s_t, s = downsample[2], downsample[3]
		self.layer_list.append(CBlock(64, 128, 1,
			conv=make_conv_layer(config, 64, 128, 1, 64, kernel_size=(k_t, 3, 3), stride=(s_t, s, s), padding='same'),
			proj=Column(MultiHeadResBlock(128, 1,
				conv=make_conv_layer(config, 128, 128, 1, 128, kernel_size=(r_k if k_t > 1 else 1, r_k, r_k), stride=1, padding='same'),
				act=act, norm=norm, order=res_order), depth=2, recurrent=rec_col),
            act=act, norm=norm, order=conv_order))
		clip_len = clip_len // s_t
		img_size = img_size // s

		in_channels = 128
		heads = 1
		for l in range(len(layer_sizes)):
			out_embed_dim, out_heads, depth = layer_sizes[l]
			in_heads = heads
			#heads = out_heads
			in_channels = (in_channels * in_heads) // heads
			out_channels = (out_embed_dim * out_heads) // heads

			# Convolutional block over spatial domain
			s = 2 if img_size > fmap_size[0] else 1
			if t_aggr == 'conv':
				s_t = 2 if clip_len > fmap_size[1] else 1
				k_t = max(patch_size, s_t)
				self.layer_list.append(CBlock(in_channels, out_channels, heads,
					conv=make_conv_layer(config, in_channels, out_channels, heads, in_channels, kernel_size=(k_t, 3, 3), stride=(s_t, s, s), padding='same'),
					proj=Column(MultiHeadResBlock(out_channels, heads,
						conv=make_conv_layer(config, out_channels, out_channels, heads, out_channels, kernel_size=r_k, stride=1, padding='same'),
						act=act, norm=norm, order=res_order), depth=depth, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
				clip_len = clip_len // s_t
			else:
				self.layer_list.append(CBlock(in_channels, out_channels, heads,
					conv=make_conv_layer(config, in_channels, out_channels, heads, in_channels, kernel_size=(1, 3, 3), stride=(1, s, s), padding='same', token_dim=(3, 4)),
					proj=Column(MultiHeadResBlock(out_channels, heads,
						conv=make_conv_layer(config, out_channels, out_channels, heads, out_channels, kernel_size=(1, r_k, r_k), stride=1, padding='same', token_dim=(3, 4)),
						act=act, norm=norm, order=res_order), depth=depth, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
			img_size = img_size // s
			#in_channels = out_channels
			in_channels = out_embed_dim
			heads = out_heads

			# Patch embedding block: convolution/pooling over temporal domain
			s_t = 2 if clip_len > fmap_size[1] else 1
			k_t = max(patch_size, s_t)
			if t_aggr == 'tconv':
				self.layer_list.append(CBlock(in_channels, in_channels, heads,
					conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)),
					proj=Column(MultiHeadResBlock(in_channels, heads,
						conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(r_k, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=depth, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
				clip_len = clip_len // s_t
			if t_aggr == 'pool':
				if s_t > 1: self.layer_list.append(nn.MaxPool3d(kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding=utils.get_padding_same((k_t, 1, 1))))
				clip_len = clip_len // s_t
			if t_aggr == 'attpool':
				self.layer_list.append(make_attentive_pool_layer(config, 1, in_channels, heads, in_channels, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)))
				clip_len = clip_len // s_t

			# Pos embedding
			if t_att is not None and ((l == 0 and pos_emb_mode == 'once') or pos_emb_mode == 'repeat'):
				self.layer_list.append(PosEmbedding(clip_len, in_channels, heads, shared_map=shared_map, token_dim=(2), cape_reg=cape_reg))
				self.layer_list.append(nn.Dropout(pos_drop))

			# Attention block over temporal domain
			if t_att is not None:
				self.layer_list.append(Column(ABlock(in_channels, heads,
				                                     attn=make_attn_layer(config, in_channels, heads, in_channels, token_dim=(2)),
				                                     proj=Column(MultiHeadResBlock(in_channels, heads,
						conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(r_k, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=(1 if t_att == 'repeat' else depth), recurrent=rec_col),
				                                     norm=norm, order=att_order), depth=(depth if t_att == 'repeat' else 1), recurrent=rec_col))

			# Aggregation block: convolution/pooling over temporal domain
			s_t = 2 if clip_len > fmap_size[1] else 1
			k_t = max(patch_size, s_t)
			if t_aggr == 'tconv_post':
				self.layer_list.append(CBlock(in_channels, in_channels, heads,
					conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)),
					proj=Column(MultiHeadResBlock(in_channels, heads,
						conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(r_k, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=depth, recurrent=rec_col),
					act=act, norm=norm, order=conv_order))
				clip_len = clip_len // s_t
			if t_aggr == 'pool_post':
				if s_t > 1: self.layer_list.append(nn.MaxPool3d(kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding=utils.get_padding_same((k_t, 1, 1))))
				clip_len = clip_len // s_t
			if t_aggr == 'attpool_post':
				self.layer_list.append(make_attentive_pool_layer(config, 1, in_channels, heads, in_channels, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)))
				clip_len = clip_len // s_t

			in_channels = out_embed_dim
			heads = out_heads

		# Aggregation and final classifier
		self.layer_list.append(make_final_pool(config, in_channels, heads))
		self.layer_list.append(norm(in_channels * heads))
		self.layer_list.append(nn.Dropout(final_drop))
		fmap_shape = utils.get_fmap_shape(self)
		self.clf = nn.Linear(utils.shape2size(fmap_shape), num_classes)

		utils.init_model_params(self, mode=init_mode)

	def get_default_input_shape(self):
		return (3, self.clip_len, self.img_size, self.img_size)

	def forward_features(self, x):
		for l in self.layer_list:
			x = l(x)
		return x

	def forward(self, x):
		if x.shape[2] != self.clip_len or x.shape[3] != self.img_size or x.shape[4] != self.img_size:
			raise RuntimeError("Input dimension ({}) does not match expected size ({})".format(x.shape[2:], (self.clip_len, self.img_size, self.img_size)))
		x = self.forward_features(x)
		x = self.clf(x.reshape(x.shape[0], -1))
		return x

	def get_train_params(self):
		return utils.disable_wd_for_pos_emb(self)

	def internal_loss(self):
		losses = [l.cape_reg_buff for l in self.layer_list if isinstance(l, PosEmbedding)]
		return sum(losses) if len(losses) > 0 else 0


def get_proj_params(channels, num_splits, heads, headsize, transposed=False, shared_splits=None):
	shared_splits = [False] * num_splits if transposed or shared_splits is None else [shared_splits[i].upper() == 'T' if i < len(shared_splits) else False for i in range(num_splits)]
	proj_splits = [(headsize if transposed else channels) * (1 if s else heads) for s in shared_splits]
	proj_heads = 1 if any(shared_splits) else heads
	proj_headsize = heads * channels if any(shared_splits) else (headsize if transposed else channels)
	proj_in = channels * (heads if any(shared_splits) else 1)
	proj_out = (channels * num_splits) if transposed else sum(proj_splits) // proj_heads
	return proj_splits, proj_in, proj_out, proj_heads, proj_headsize

def make_conv_layer(config, in_channels, out_channels, heads, headsize, kernel_size=3, stride=1, padding=0, token_dim=None, transposed=False):
	conv_type = config.get('conv_type', 'MultiHeadConv3d')
	shared_map = config.get('shared_map', False)
	shared_qkv = config.get('shared_qkv', None)
	norm = utils.retrieve(config.get('norm', 'models.modutils.BatchNorm'))

	conv = None
	if conv_type == 'MultiHeadConv3d':
		conv = MultiHeadConv3d(in_channels, out_channels, heads, headsize, shared_map=shared_map,
							   kernel_size=kernel_size, stride=stride, padding=padding, token_dim=token_dim, transposed=transposed)
	if conv_type == 'AttentiveConv3d':
		if transposed and out_channels % in_channels != 0:
			raise ValueError("Out channels ({}) must be multiple of in channels ({}) in transpose mode".format(out_channels, in_channels))
		in_headsize = headsize if transposed else in_channels
		out_headsize = headsize * out_channels // in_channels if transposed else out_channels
		qkv_splits, *qkv_params = get_proj_params(shared_qkv, in_channels, AttentiveConv3d.QKV_SPLITS, heads, in_headsize, transposed)
		qkv_b_splits, *qkv_b_params = get_proj_params(shared_qkv, in_channels, AttentiveConv3d.QKV_SPLITS, heads, out_headsize, transposed)
		conv = AttentiveConv3d(in_channels, out_channels, heads, headsize, shared_map=shared_map,
			kernel_size=kernel_size, stride=stride, padding=padding, token_dim=token_dim, transposed=transposed,
			qkv=make_proj_layer(config, *qkv_params, token_dim=token_dim, transposed=transposed, splits=qkv_splits), qkv_splits=qkv_splits,
			qkv_b=make_proj_layer(config, *qkv_b_params, token_dim=token_dim, transposed=transposed, splits=qkv_b_splits), qkv_b_splits=qkv_b_splits)
	if conv_type == 'AttentionConv3dBlock':
		attn = make_attn_layer(config, in_channels, heads, headsize, token_dim=token_dim, transposed=transposed)
		conv = make_conv_layer(
			{k: v  if k != 'conv_type' else 'MultiHeadConv3d' for k, v in config.items()},
			in_channels, out_channels, heads, headsize,
			kernel_size=kernel_size, stride=stride, padding=padding, token_dim=token_dim, transposed=transposed)
		conv = AttentionConv3dBlock(in_channels, heads, attn=attn, norm=norm, conv=conv)

	if conv is None:
		raise ValueError("Convolution type {} not supported".format(conv_type))

	return conv

def make_attn_layer(config, channels, heads, headsize, token_dim=None, transposed=False):
	attn_type = config.get('attn_type', 'Attention')
	attn_hidden_rank = config.get('attn_hidden_rank', 64)
	attn_kernel_size = config.get('attn_kernel_size', 3)
	shared_qkv = config.get('shared_qkv', None)
	drop = config.get('drop', 0.)
	attn_proj = config.get('attn_proj', False)
	proj_splits, *proj_params = get_proj_params(channels, 1, heads, headsize, transposed)
	proj = make_proj_layer(config, *proj_params, token_dim=token_dim, transposed=transposed, splits=proj_splits) if attn_proj else None

	attn = None
	if attn_type == 'Attention':
		qkv_splits, *qkv_params = get_proj_params(channels, Attention.QKV_SPLITS, heads, headsize, transposed, shared_qkv)
		attn = Attention(headsize if transposed else channels, heads, token_dim=token_dim, transposed=transposed,
			qkv=make_proj_layer(config, *qkv_params, token_dim=token_dim, transposed=transposed, splits=qkv_splits), qkv_splits=qkv_splits,
			proj=proj, attn_drop=drop, proj_drop=drop)
	if attn_type == 'KernelAttention':
		qkv_splits, *qkv_params = get_proj_params(channels, KernelAttention.QKV_SPLITS, heads, headsize, transposed, shared_qkv)
		attn = KernelAttention(headsize if transposed else channels, heads, features=channels, knl_size=attn_hidden_rank, token_dim=token_dim, transposed=transposed,
			qkv=make_proj_layer(config, *qkv_params, token_dim=token_dim, transposed=transposed, splits=qkv_splits), qkv_splits=qkv_splits,
			proj=proj, attn_drop=drop, proj_drop=drop)
	if attn_type == 'HiddenRankAttention':
		qkv_splits, *qkv_params = get_proj_params(channels, HiddenRankAttention.QKV_SPLITS, heads, headsize, transposed, shared_qkv)
		attn = HiddenRankAttention(headsize if transposed else channels, heads, features=channels, hidden_rank=attn_hidden_rank, token_dim=token_dim, transposed=transposed,
			qkv=make_proj_layer(config, *qkv_params, token_dim=token_dim, transposed=transposed, splits=qkv_splits), qkv_splits=qkv_splits,
			proj=proj, attn_drop=drop, proj_drop=drop)
	if attn_type == 'LocalAttention':
		qkv_splits, *qkv_params = get_proj_params(channels, KernelAttention.QKV_SPLITS, heads, headsize, transposed, shared_qkv)
		attn = LocalAttention(headsize if transposed else channels, heads, token_dim=token_dim, transposed=transposed,
            kernel_size=attn_kernel_size, stride=1, padding='same',
			qkv=make_proj_layer(config, *qkv_params, token_dim=token_dim, transposed=transposed, splits=qkv_splits), qkv_splits=qkv_splits,
			proj=proj, attn_drop=drop, proj_drop=drop)

	if attn is None:
		raise ValueError("Attention type {} not supported".format(attn_type))

	return attn

def make_proj_layer(config, in_channels, out_channels, heads, headsize, token_dim=None, transposed=False, splits=None):
	proj_map_type = config.get('proj_map_type', 'MultiHeadLinear')
	shared_map = config.get('shared_map', False)
	proj_kernel_size = config.get('proj_kernel_size', 3)
	proj_hidden_layers = config.get('proj_hidden_layers', 1)
	conv_order = config.get('conv_order', 'NCNAP')
	res_order = config.get('res_order', 'CNACNSA')
	res_kernel_size = config.get('res_kernel_size', 3)
	rec_col = config.get('rec_col', False),
	act = utils.retrieve(config.get('act', 'torch.nn.ReLU'))
	norm = utils.retrieve(config.get('norm', 'models.modutils.BatchNorm'))
	drop = config.get('drop', 0.)

	proj = None
	if proj_map_type == 'MultiHeadLinear':
		proj = MultiHeadLinear(in_channels, out_channels, heads, headsize, shared_map=shared_map,
			token_dim=token_dim, transposed=transposed)
	if proj_map_type == 'MultiHeadConv3d':
		proj = MultiHeadConv3d(in_channels, out_channels, heads, headsize, shared_map=shared_map,
			kernel_size=proj_kernel_size, stride=1, padding='same', token_dim=token_dim, transposed=transposed)
	if proj_map_type == 'MultiHeadMlp':
		proj = MultiHeadMLP(in_channels, out_channels, out_channels, heads, headsize, shared_map=shared_map,
			hidden_layers=proj_hidden_layers, recurrent=False, act=act, norm=norm, token_dim=token_dim,
			transposed=transposed, drop=drop)
	if proj_map_type == 'CBlock':
		proj = CBlock(in_channels, out_channels, heads,
			conv=MultiHeadConv3d(in_channels, out_channels, heads, headsize, shared_map=shared_map,
				kernel_size=proj_kernel_size, stride=1, padding='same', token_dim=token_dim, transposed=transposed),
			proj=Column(MultiHeadResBlock(out_channels, heads,
				conv=MultiHeadConv3d(out_channels, out_channels, heads, headsize, shared_map=shared_map,
					kernel_size=res_kernel_size, stride=1, padding='same', token_dim=token_dim, transposed=transposed),
				act=act, norm=norm, order=res_order, drop=drop), depth=proj_hidden_layers, recurrent=rec_col),
			act=act, norm=norm, order=conv_order, splits=splits)

	if proj is None:
		raise ValueError("Proj map type {} not supported".format(proj_map_type))

	return proj

def make_attentive_pool_layer(config, size, channels, heads, headsize, kernel_size=3, stride=1, padding=0, token_dim=None, transposed=False):
	shared_map = config.get('shared_map', False)
	shared_qkv = config.get('shared_qkv', None)
	drop = config.get('drop', 0.)
	attn_proj = config.get('attn_proj', False)
	proj_splits, *proj_params = get_proj_params(channels, 1, heads, headsize, transposed)
	proj = make_proj_layer(config, *proj_params, token_dim=token_dim, transposed=transposed, splits=proj_splits) if attn_proj else None
	qkv_splits, *qkv_params = get_proj_params(channels, AttentivePool3d.QKV_SPLITS, heads, headsize, transposed=transposed, shared_splits=shared_qkv)
	qkv = make_proj_layer(config, *qkv_params, token_dim=token_dim, transposed=transposed, splits=qkv_splits)

	return AttentivePool3d(size, headsize if transposed else channels, heads, shared_map=shared_map, kernel_size=kernel_size, stride=stride, padding=padding,
		token_dim=token_dim, transposed=transposed, qkv=qkv, qkv_splits=qkv_splits, proj=proj,  attn_drop=drop, proj_drop=drop)

def make_final_pool(config, channels, heads):
	token_pool = config.get('token_pool', 'avg')

	final_pool = None
	if token_pool == 'avg':
		final_pool = nn.AdaptiveAvgPool3d(1)
	if token_pool == 'max':
		final_pool = nn.AdaptiveMaxPool3d(1)
	if token_pool == 'attentive':
		final_pool = make_attentive_pool_layer(config, 1, channels, heads, channels, kernel_size=-1, stride=-1)

	if final_pool is None:
		raise ValueError("Pool Type {} not supported".format(token_pool))

	return final_pool


if __name__ == "__main__":
	import torch
	config = {'layer_sizes': ((64, 4, 2), (64, 8, 2))}
	inputs = torch.rand(1, 3, 16, 112, 112)
	net = CA3D(config, 101)

	outputs = net(inputs)
	print(outputs.size())
import torch
import torch.nn as nn

from .modutils import *
from .CA3D import get_proj_params, make_conv_layer, make_attn_layer, make_proj_layer, make_attentive_pool_layer, make_final_pool
import utils

class CA3D(nn.Module):
	def __init__(self, config, num_classes):
		super().__init__()

		clip_len = config.get('clip_len', 32)
		img_size = config.get('input_size', 112)
		fmap_size = config.get('fmap_size', (1, 1))
		patch_size = config.get('patch_size', 3)
		layer_sizes = config.get('layer_sizes', ((64, 4, 2), (64, 8, 2)))
		conv_order = config.get('conv_order', 'CANPN')
		res_order = config.get('res_order', 'CANCANS')
		att_order = config.get('att_order', 'NASNP')
		r_k = config.get('res_kernel_size', 3)
		rec_col = config.get('rec_col', False)
		shared_map = config.get('shared_map', False)
		cape_reg = config.get('cape_reg', 0)
		act = utils.retrieve(config.get('act', 'torch.nn.ReLU'))()
		norm = utils.retrieve(config.get('norm', 'models.modutils.BatchNorm'))
		attn_norm = utils.retrieve(config.get('attn_norm', config.get('norm', 'models.modutils.BatchNorm')))
		downsample = config.get('downsample', (1, 2, 1, 2, 1, 1))
		t_aggr = config.get('t_aggr', 'conv')
		t_att = config.get('t_att', 'once')
		init_mode = config.get('init_mode', 'kaiming_normal')
		pos_emb_mode = config.get('pos_emb_mode', 'once')
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
		k_t = downsample[4]
		self.layer_list.append(CBlock(3, 64, 1,
			conv=make_conv_layer(config, 3, 64, 1, 3, kernel_size=(k_t, 7, 7), stride=(s_t if k_t > 1 else 1, s, s), padding='same'),
			proj=Column(MultiHeadResBlock(64, 1,
				conv=make_conv_layer(config, 64, 64, 1, 64, kernel_size=(r_k if k_t > 1 else 1, r_k, r_k), stride=1, padding='same'),
				act=act, norm=norm, order=res_order), depth=0, recurrent=rec_col),
            act=act, norm=norm, order='CNPNA'))
		if k_t == 1 and s_t > 1:
			if t_aggr in ['tconv', 'tconv_post']:
				self.layer_list.append(CBlock(64, 64, 1,
					conv=make_conv_layer(config, 64, 64, 1, 64, kernel_size=(s_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)),
					proj=Column(MultiHeadResBlock(64, 1,
						conv=make_conv_layer(config, 64, 64, 1, 64, kernel_size=(1, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=0, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
			else: self.layer_list.append(nn.MaxPool3d(kernel_size=(s_t, 1, 1), stride=(s_t, 1, 1), padding=utils.get_padding_same((s_t, 1, 1))))
		clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')
		img_size = utils.get_conv_output_size(img_size, 3, s, padding='same')

		# Second layer: spatio-temporal convolutional layer with residual column
		s_t, s = downsample[2], downsample[3]
		k_t = downsample[5]
		self.layer_list.append(CBlock(64, 128, 1,
			conv=make_conv_layer(config, 64, 128, 1, 64, kernel_size=(k_t, 3, 3), stride=(s_t if k_t > 1 else 1, s, s), padding='same'),
			proj=Column(MultiHeadResBlock(128, 1,
				conv=make_conv_layer(config, 128, 128, 1, 128, kernel_size=(r_k if k_t > 1 else 1, r_k, r_k), stride=1, padding='same'),
				act=act, norm=norm, order=res_order), depth=2, recurrent=rec_col),
            act=act, norm=norm, order=conv_order))
		if k_t == 1 and s_t > 1:
			if t_aggr in ['tconv', 'tconv_post']:
				self.layer_list.append(CBlock(128, 128, 1,
					conv=make_conv_layer(config, 128, 128, 1, 128, kernel_size=(s_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)),
					proj=Column(MultiHeadResBlock(128, 1,
						conv=make_conv_layer(config, 128, 128, 1, 128, kernel_size=(1, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=0, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
			else: self.layer_list.append(nn.MaxPool3d(kernel_size=(s_t, 1, 1), stride=(s_t, 1, 1), padding=utils.get_padding_same((s_t, 1, 1))))
		clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')
		img_size = utils.get_conv_output_size(img_size, 3, s, padding='same')

		in_channels = 128
		heads = 1
		for l in range(len(layer_sizes)):
			out_embed_dim, out_heads, depth = layer_sizes[l]
			in_heads = heads
			#heads = out_heads
			in_channels = (in_channels * in_heads) // heads
			out_channels = (out_embed_dim * out_heads) // heads

			# Convolutional block over spatial domain
			s = 2 if img_size > self.fmap_size[0] else 1
			if t_aggr == 'conv':
				s_t = 2 if clip_len > self.fmap_size[1] else 1
				k_t = max(patch_size, s_t)
				self.layer_list.append(CBlock(in_channels*heads, out_channels*heads, 1,
					conv=make_conv_layer(config, in_channels*heads, out_channels*heads, 1, in_channels*heads, kernel_size=(k_t, 3, 3), stride=(s_t, s, s), padding='same'),
					proj=Column(MultiHeadResBlock(out_channels, heads,
						conv=make_conv_layer(config, out_channels, out_channels, heads, out_channels, kernel_size=r_k, stride=1, padding='same'),
						act=act, norm=norm, order=res_order), depth=depth, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
				clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')
			else:
				self.layer_list.append(CBlock(in_channels*heads, out_channels*heads, 1,
					conv=make_conv_layer(config, in_channels*heads, out_channels*heads, 1, in_channels*heads, kernel_size=(1, 3, 3), stride=(1, s, s), padding='same', token_dim=(3, 4)),
					proj=Column(MultiHeadResBlock(out_channels, heads,
						conv=make_conv_layer(config, out_channels, out_channels, heads, out_channels, kernel_size=(1, r_k, r_k), stride=1, padding='same', token_dim=(3, 4)),
						act=act, norm=norm, order=res_order), depth=depth, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
			img_size = utils.get_conv_output_size(img_size, 3, s, padding='same')
			#in_channels = out_channels
			in_channels = out_embed_dim
			heads = out_heads

			# Patch embedding block: convolution/pooling over temporal domain
			s_t = 2 if clip_len > self.fmap_size[1] else 1
			k_t = max(patch_size, s_t)
			if t_aggr == 'tconv':
				self.layer_list.append(CBlock(in_channels*heads, in_channels*heads, 1,
					conv=make_conv_layer(config, in_channels*heads, in_channels*heads, 1, in_channels*heads, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)),
					proj=Column(MultiHeadResBlock(in_channels, heads,
						conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(r_k, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=0, recurrent=rec_col),
		            act=act, norm=norm, order=conv_order))
				clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')
			if t_aggr == 'pool':
				if s_t > 1: self.layer_list.append(nn.MaxPool3d(kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding=utils.get_padding_same((k_t, 1, 1))))
				clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')
			if t_aggr == 'attpool':
				self.layer_list.append(make_attentive_pool_layer(config, 1, in_channels, heads, in_channels, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)))
				clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')

			# Pos embedding
			if t_att is not None and ((l == 0 and pos_emb_mode == 'once') or pos_emb_mode == 'repeat'):
				self.layer_list.append(PosEmbedding(clip_len, in_channels*heads, 1, shared_map=shared_map, token_dim=(2), cape_reg=cape_reg))
				self.layer_list.append(nn.Dropout(pos_drop))

			# Attention block over temporal domain
			if t_att is not None:
				self.layer_list.append(Column(ABlock(in_channels, heads,
				                                     attn=make_attn_layer(config, in_channels, heads, in_channels, token_dim=(2)),
				                                     proj=Column(MultiHeadResBlock(in_channels, heads,
						conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(r_k, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=(1 if t_att == 'repeat' else depth), recurrent=rec_col),
				                                     norm=attn_norm, order=att_order), depth=(depth if t_att == 'repeat' else 1), recurrent=rec_col))

			# Aggregation block: convolution/pooling over temporal domain
			s_t = 2 if clip_len > self.fmap_size[1] else 1
			k_t = max(patch_size, s_t)
			if t_aggr == 'tconv_post':
				self.layer_list.append(CBlock(in_channels*heads, in_channels*heads, 1,
					conv=make_conv_layer(config, in_channels*heads, in_channels*heads, 1, in_channels*heads, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)),
					proj=Column(MultiHeadResBlock(in_channels, heads,
						conv=make_conv_layer(config, in_channels, in_channels, heads, in_channels, kernel_size=(r_k, 1, 1), stride=1, padding='same', token_dim=(2)),
						act=act, norm=norm, order=res_order), depth=0, recurrent=rec_col),
					act=act, norm=norm, order=conv_order))
				clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')
			if t_aggr == 'pool_post':
				if s_t > 1: self.layer_list.append(nn.MaxPool3d(kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding=utils.get_padding_same((k_t, 1, 1))))
				clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')
			if t_aggr == 'attpool_post':
				self.layer_list.append(make_attentive_pool_layer(config, 1, in_channels, heads, in_channels, kernel_size=(k_t, 1, 1), stride=(s_t, 1, 1), padding='same', token_dim=(2)))
				clip_len = utils.get_conv_output_size(clip_len, k_t, s_t, padding='same')

			in_channels = out_embed_dim
			heads = out_heads

		# Aggregation and final classifier
		self.layer_list.append(make_final_pool(config, in_channels*heads, 1))
		self.layer_list.append(norm(in_channels * heads))
		self.layer_list.append(nn.Dropout(final_drop))
		fmap_shape = utils.get_fmap_shape(self)
		self.clf = nn.Linear(utils.shape2size(fmap_shape), num_classes)

		utils.init_model_params(self, mode=init_mode)

	def get_default_input_shape(self):
		return (3, self.clip_len, self.img_size, self.img_size)

	def forward_features(self, x):
		for i, l in enumerate(self.layer_list):
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



if __name__ == "__main__":
	import torch
	config = {'layer_sizes': ((64, 4, 2), (64, 8, 2))}
	inputs = torch.rand(1, 3, 16, 112, 112)
	net = CA3D(config, 101)

	outputs = net(inputs)
	print(outputs.size())
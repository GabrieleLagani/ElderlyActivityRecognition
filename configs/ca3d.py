from .defaults import *

ca3d = {}
ca3d_1 = {}

for d, p, lr in [(d, p, lr) for d in datasets for p in precisions for lr in lrs]:

	ca3d[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.CA3D.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
	    'downsample': (1, 2, 1, 2, 1, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'CANCANS', 'rec_col': False,
		'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 1, 'headwise_map': True, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'attn_norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': False, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.5,
	    **precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	ca3d['l_' + p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.CA3D.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
	    'downsample': (1, 2, 1, 2, 1, 1), 'layer_sizes': ((64, 4, 4), (64, 8, 8)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'CANCANS', 'rec_col': False,
		'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 1, 'headwise_map': True, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': False, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.5,
	    **precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	ca3d['lr_' + p + '_' + lr + '_' + d] = {
		**data_defaults[d],
		'model': 'models.CA3D_reduced.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
		'downsample': (1, 2, 1, 2, 1, 1), 'layer_sizes': ((64, 4, 4), (64, 8, 8)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'CANCANS', 'rec_col': False,
		'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 1, 'headwise_map': True, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'attn_norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': False, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5,
		**sched_params[d],
	}

	ca3d['lrc_' + p + '_' + lr + '_' + d] = {
		**data_defaults[d],
		'model': 'models.CA3D_reduced.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
		'downsample': (2, 2, 2, 2, 3, 3), 'layer_sizes': ((64, 4, 4), (64, 8, 8)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'CANCANS', 'rec_col': False,
		'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 1, 'headwise_map': True, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'attn_norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': False, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5,
		**sched_params[d],
	}

	ca3d['lf_' + p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.CA3D_fullconv.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
	    'downsample': (1, 2, 1, 2, 1, 1), 'layer_sizes': ((64, 4, 4), (64, 8, 8)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'CANCANS', 'rec_col': False,
		'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 1, 'headwise_map': True, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': False, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.5,
	    **precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	ca3d_1[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.CA3D.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
	    'downsample': (1, 2, 1, 2, 1, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'CANCANS', 'rec_col': False,
		'attn_type': 'KernelAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 1, 'headwise_map': True, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': False, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.5,
	    **precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}



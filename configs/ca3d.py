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
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
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
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
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
		'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5,
		'epochs': 150, 'sched_milestones': range(50, 150, 10), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
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
		'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5,
		'epochs': 150, 'sched_milestones': range(50, 150, 10), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
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
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
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
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
	}



from .defaults import *

chi_params = {
	#'st': {'enc_kernel_sizes': ((21, 7, 7), (1, 21, 21)), 'enc_strides': ((8, 2, 2), (1, 8, 8)), 'enc_channels': (64, 64), 'fmap_size': (4, 4), 'shrink_min_dim': True, 'token_dim': (2, 3, 4), 'x_dim': ('s', 't'), 'alternate_attn': False, 'convattn': False},
	'st': {'enc_kernel_sizes': ((21, 7, 7), (1, 35, 35)), 'enc_strides': ((8, 2, 2), (1, 16, 16)), 'enc_channels': (64, 64), 'fmap_size': (4, 4), 'shrink_min_dim': True, 'token_dim': (2, 3, 4), 'x_dim': ('s', 't'), 'alternate_attn': False, 'convattn': True},
	#'sc': {'enc_kernel_sizes': ((1, 7, 7), (1, 21, 21)), 'enc_strides': ((1, 2, 2), (1, 8, 8)), 'enc_channels': (8, 64), 'fmap_size': (8, 4), 'shrink_min_dim': True, 'token_dim': (2, 3, 4), 'x_dim': ('s', 'c'), 'alternate_attn': False, 'convattn': False},
	'sc': {'enc_kernel_sizes': ((1, 7, 7), (1, 35, 35)), 'enc_strides': ((1, 2, 2), (1, 16, 16)), 'enc_channels': (8, 64), 'fmap_size': (8, 4), 'shrink_min_dim': True, 'token_dim': (2, 3, 4), 'x_dim': ('s', 'c'), 'alternate_attn': False, 'convattn': True},
	#'tc': {'enc_kernel_sizes': ((1, 7, 7), (21, 7, 7)), 'enc_strides': ((1, 2, 2), (8, 2, 2)), 'enc_channels': (8, 64), 'fmap_size': (4, 4), 'shrink_min_dim': True, 'token_dim': (2, 3, 4), 'x_dim': ('t', 'c'), 'alternate_attn': False, 'convattn': False},
	'tc': {'enc_kernel_sizes': ((1, 7, 7), (21, 7, 7)), 'enc_strides': ((1, 2, 2), (8, 2, 2)), 'enc_channels': (8, 64), 'fmap_size': (4, 4), 'shrink_min_dim': True, 'token_dim': (2, 3, 4), 'x_dim': ('t', 'c'), 'alternate_attn': False, 'convattn': True},

	#'st_reduced': {'enc_kernel_sizes': ((21, 7, 7), (1, 21, 21)), 'enc_strides': ((8, 2, 2), (1, 8, 8)), 'enc_channels': (64, 64), 'fmap_size': (8, 4), 'shrink_min_dim': False, 'token_dim': (2, 3, 4), 'x_dim': ('s', 't'), 'alternate_attn': False, 'convattn': False},
	'st_reduced': {'enc_kernel_sizes': ((21, 7, 7), (1, 35, 35)), 'enc_strides': ((8, 2, 2), (1, 16, 16)), 'enc_channels': (64, 64), 'fmap_size': (8, 4), 'shrink_min_dim': False, 'token_dim': (2, 3, 4), 'x_dim': ('s', 't'), 'alternate_attn': False, 'convattn': True},
	#'sc_reduced': {'enc_kernel_sizes': ((1, 7, 7), (1, 21, 21)), 'enc_strides': ((1, 2, 2), (1, 8, 8)), 'enc_channels': (8, 64), 'fmap_size': (8, 4), 'shrink_min_dim': False, 'token_dim': (2, 3, 4), 'x_dim': ('s', 'c'), 'alternate_attn': False, 'convattn': False},
	'sc_reduced': {'enc_kernel_sizes': ((1, 7, 7), (1, 35, 35)), 'enc_strides': ((1, 2, 2), (1, 16, 16)), 'enc_channels': (8, 64), 'fmap_size': (8, 4), 'shrink_min_dim': False, 'token_dim': (2, 3, 4), 'x_dim': ('s', 'c'), 'alternate_attn': False, 'convattn': True},
	#'tc_reduced': {'enc_kernel_sizes': ((1, 7, 7), (21, 7, 7)), 'enc_strides': ((1, 2, 2), (8, 2, 2)), 'enc_channels': (8, 64), 'fmap_size': (8, 4), 'shrink_min_dim': False, 'token_dim': (2, 3, 4), 'x_dim': ('t', 'c'), 'alternate_attn': False, 'convattn': False},
	'tc_reduced': {'enc_kernel_sizes': ((1, 7, 7), (21, 7, 7)), 'enc_strides': ((1, 2, 2), (8, 2, 2)), 'enc_channels': (8, 64), 'fmap_size': (8, 4), 'shrink_min_dim': False, 'token_dim': (2, 3, 4), 'x_dim': ('t', 'c'), 'alternate_attn': False, 'convattn': True},

}

schinet = {}

for d, p, lr, chi_mode in [(d, p, lr, chi_mode) for d in datasets for p in precisions for lr in lrs for chi_mode in chi_params]:
	schinet[chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 2), (*chi_params[chi_mode]['enc_channels'], 8, 2)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['chi_conv_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_chi_conv.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 2), (*chi_params[chi_mode]['enc_channels'], 8, 2)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['chi_mixed_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_chi_mixed.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 2), (*chi_params[chi_mode]['enc_channels'], 8, 2)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['path_2plus1_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_path_2plus1.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 2), (*chi_params[chi_mode]['enc_channels'], 8, 2)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['conv_2plus1_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_conv_2plus1.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 2), (*chi_params[chi_mode]['enc_channels'], 8, 2)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['xl_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 4), (*chi_params[chi_mode]['enc_channels'], 4, 8), (*chi_params[chi_mode]['enc_channels'], 8, 16)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['chi_conv_xl_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_chi_conv.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 4), (*chi_params[chi_mode]['enc_channels'], 4, 8), (*chi_params[chi_mode]['enc_channels'], 8, 16)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['chi_mixed_xl_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_chi_mixed.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 4), (*chi_params[chi_mode]['enc_channels'], 4, 8), (*chi_params[chi_mode]['enc_channels'], 8, 16)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['path_2plus1_xl_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_path_2plus1.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 4), (*chi_params[chi_mode]['enc_channels'], 4, 8), (*chi_params[chi_mode]['enc_channels'], 8, 16)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	schinet['conv_2plus1_xl_' + chi_mode + '_' + p + '_' + lr + '_' + d] = {
		**data_defaults_large_clip[d],
		'model': 'models.SChiNet_conv_2plus1.SChiNet',
		**chi_params[chi_mode],
		'chi_stages': ((*chi_params[chi_mode]['enc_channels'], 4, 4), (*chi_params[chi_mode]['enc_channels'], 4, 8), (*chi_params[chi_mode]['enc_channels'], 8, 16)),
		'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		**precisions[p],
		'batch_size': 32, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}



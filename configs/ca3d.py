ca3d_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.CA3D.CA3D',
    'clip_len': 16, 'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
    'downsample': (1, 2, 2, 2, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'pool', 't_att': 'once',
	'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'ANCANCS', 'rec_col': False,
	'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
	'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 3, 'shared_map': False, 'shared_qkv': 'FFF',
	'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
	'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': True, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
    'precision': 'float32', 'stretch': (1, 1),
	'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-3,
	'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
	#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
}

ca3d_ucf1 = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.CA3D.CA3D',
    'clip_len': 16, 'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
    'downsample': (1, 2, 2, 2, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'pool', 't_att': 'once',
	'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'ANCANCS', 'rec_col': False,
	'attn_type': 'KernelAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
	'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 3, 'shared_map': False, 'shared_qkv': 'FFF',
	'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
	'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': True, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
    'precision': 'float32', 'stretch': (1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-3,
	'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
	#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
}


ca3d_ucf_relu = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.CA3D.CA3D',
    'clip_len': 16, 'patch_size': 3, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
    'downsample': (1, 2, 2, 2, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'conv', 't_att': 'once',
	'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCNAP', 'res_order': 'NACNACS', 'rec_col': False,
	'attn_type': 'Attention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
	'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 3, 'shared_map': False, 'shared_qkv': 'FFF',
	'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
	'pos_emb_mode': 'repeat', 'disable_wd_for_pos_emb': True, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
    'precision': 'float32', 'stretch': (1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-3,
	'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
	#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
}

ca3d_ucf_elu = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.CA3D.CA3D',
    'clip_len': 16, 'patch_size': 3, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
    'downsample': (1, 2, 2, 2, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'conv', 't_att': 'once',
	'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCNAP', 'res_order': 'NACNACS', 'rec_col': False,
	'attn_type': 'Attention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
	'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 3, 'shared_map': False, 'shared_qkv': 'FFF',
	'act': 'torch.nn.ELU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'trunc_normal',
	'pos_emb_mode': 'repeat', 'disable_wd_for_pos_emb': True, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
    'precision': 'float32', 'stretch': (1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4,
	'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
	#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
}

ca3d_ucf_mish = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.CA3D.CA3D',
    'clip_len': 16, 'patch_size': 3, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
    'downsample': (1, 2, 2, 2, 1), 'st_emb': True, 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'conv', 't_att': 'once',
	'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCNAP', 'res_order': 'NACNACS', 'rec_col': False,
	'attn_type': 'Attention', 'attn_hidden_rank': 64, 'attn_kernel_size': 3, 'att_order': 'NASNP', 'attn_proj': False,
	'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 3, 'shared_map': False, 'shared_qkv': 'FFF',
	'act': 'torch.nn.Mish', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'trunc_normal',
	'pos_emb_mode': 'repeat', 'disable_wd_for_pos_emb': True, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
    'precision': 'float32', 'stretch': (1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4,
	'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
	#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
}


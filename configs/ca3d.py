datasets = {'ucf101': 'dataloaders.videodataset.UCF101DataManager',
            'hmdb51': 'dataloaders.videodataset.HMDB51DataManager', 
            'kinetics400': 'dataloaders.videodataset.KineticsDataManager'}

precisions = {'f16': 'float16', 'f16_qat': 'float16', 'f32': 'float32'}

ca3d = {}
ca3d_1 = {}

for d, p in [(d, p) for d in datasets for p in precisions]:

	ca3d[p + '_' + d] = {
	    'data_manager': datasets[d],
		'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
		'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
		'frame_resize': 112, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
		'frames_per_clip': 20, 'space_between_frames': 1, 'min_space_between_frames': None, 'max_space_between_frames': None,
		'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
		'eval_frames_per_clip': 20, 'eval_space_between_frames': 1, 'eval_auto_frames': None,
		'input_size': 112, 'da_time_scale_rel_delta': 0.,
		'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
		'auto_len': None, 'min_auto_len': None, 'max_auto_len': None,
		'eval_clip_len': 16, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
	    'model': 'models.CA3D.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
	    'downsample': (1, 2, 2, 2, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'ANCANCS', 'rec_col': False,
		'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 3, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': True, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
	    'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4 if d.startswith('kinetics') else 5e-3,
		'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
	}

	ca3d_1[p + '_' + d] = {
	    'data_manager': datasets[d],
		'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
		'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
		'frame_resize': 112, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
		'frames_per_clip': 20, 'space_between_frames': 1, 'min_space_between_frames': None, 'max_space_between_frames': None,
		'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
		'eval_frames_per_clip': 20, 'eval_space_between_frames': 1, 'eval_auto_frames': None,
		'input_size': 112, 'da_time_scale_rel_delta': 0.,
		'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
		'auto_len': None, 'min_auto_len': None, 'max_auto_len': None,
		'eval_clip_len': 16, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
	    'model': 'models.CA3D.CA3D',
		'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
	    'downsample': (1, 2, 2, 2, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'ANCANCS', 'rec_col': False,
		'attn_type': 'KernelAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 3, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': True, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
	    'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4 if d.startswith('kinetics') else 5e-3,
		'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
	}



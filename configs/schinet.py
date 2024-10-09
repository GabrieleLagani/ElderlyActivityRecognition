from .defaults import *

schinet = {}

for d, p, lr in [(d, p, lr) for d in datasets for p in precisions for lr in lrs]:

	schinet['v1_t_st_' + p + '_' + lr + '_' + d] = {
		# kinetics400 | f32 53%, 5.9GB | f16 49%, 3.2GB
	    **data_defaults[d],
		'frame_resize': 112, 'frame_resample': 1, 'input_size': 112,
		'frames_per_clip': 40, 'space_between_frames': 2,
		'eval_frames_per_clip': 40, 'eval_space_between_frames': 2,
		'clip_len': 32, 'clip_location': 'random', 'clip_step': 1,
		'eval_clip_len': 32, 'eval_clip_location': 'center', 'eval_clip_step': 1,
	    'model': 'models.SChiNet.SChiNet',
		'enc_kernel_sizes': ((11, 7, 7), (3, 35, 35)), 'enc_strides': ((4, 2, 2), (2, 16, 16)), 'enc_channels': (64, 64),
	    'chi_stages': ((64, 64, 4, 2), (64, 64, 8, 2)), 'fmap_size': (3, 3), 'head_regroup_post': False,
		'token_dim': (2, 3, 4), 'x_dim': (3, 2), 'res_kernel_size': 3, 'shared_map': False, 'fullconv': False,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
	}

	schinet['v1_t_sc_' + p + '_' + lr + '_' + d] = {
		# kinetics400 |  |
	    **data_defaults[d],
		'frame_resize': 112, 'frame_resample': 1, 'input_size': 112,
		'frames_per_clip': 40, 'space_between_frames': 2,
		'eval_frames_per_clip': 40, 'eval_space_between_frames': 2,
		'clip_len': 32, 'clip_location': 'random', 'clip_step': 1,
		'eval_clip_len': 32, 'eval_clip_location': 'center', 'eval_clip_step': 1,
	    'model': 'models.SChiNet.SChiNet',
		'enc_kernel_sizes': ((3, 7, 7), (3, 35, 35)), 'enc_strides': ((2, 2, 2), (2, 16, 16)), 'enc_channels': (16, 64),
	    'chi_stages': ((16, 64, 4, 2), (16, 64, 8, 2)), 'fmap_size': (3, 3), 'head_regroup_post': False,
		'token_dim': (2, 3, 4), 'x_dim': (3, 4), 'res_kernel_size': 3, 'shared_map': False, 'fullconv': False,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
	}

	schinet['v1_t_tc_' + p + '_' + lr + '_' + d] = {
		# kinetics400 |  |
		**data_defaults[d],
		'frame_resize': 112, 'frame_resample': 1, 'input_size': 112,
		'frames_per_clip': 40, 'space_between_frames': 2,
		'eval_frames_per_clip': 40, 'eval_space_between_frames': 2,
		'clip_len': 32, 'clip_location': 'random', 'clip_step': 1,
		'eval_clip_len': 32, 'eval_clip_location': 'center', 'eval_clip_step': 1,
		'model': 'models.SChiNet.SChiNet',
		'enc_kernel_sizes': ((3, 7, 7), (11, 7, 7)), 'enc_strides': ((2, 2, 2), (4, 2, 2)), 'enc_channels': (16, 64),
		'chi_stages': ((16, 64, 4, 2), (16, 64, 8, 2)), 'fmap_size': (3, 3), 'head_regroup_post': False,
		'token_dim': (2, 3, 4), 'x_dim': (2, 4), 'res_kernel_size': 3, 'shared_map': False, 'fullconv': False,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
		'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50,
		'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		# 'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
	}

	schinet['v1_b_st_' + p + '_' + lr + '_' + d] = {
		# kinetics400 |  |
	    **data_defaults[d],
		'frame_resize': 112, 'frame_resample': 1, 'input_size': 112,
		'frames_per_clip': 40, 'space_between_frames': 2,
		'eval_frames_per_clip': 40, 'eval_space_between_frames': 2,
		'clip_len': 32, 'clip_location': 'random', 'clip_step': 1,
		'eval_clip_len': 32, 'eval_clip_location': 'center', 'eval_clip_step': 1,
	    'model': 'models.SChiNet.SChiNet',
		'enc_kernel_sizes': ((11, 7, 7), (3, 35, 35)), 'enc_strides': ((4, 2, 2), (2, 16, 16)), 'enc_channels': (64, 64),
	    'chi_stages': ((64, 64, 4, 2), (64, 64, 4, 2), (64, 64, 8, 2), (64, 64, 8, 2)), 'fmap_size': (3, 3), 'head_regroup_post': False,
		'token_dim': (2, 3, 4), 'x_dim': (3, 2), 'res_kernel_size': 3, 'shared_map': False, 'fullconv': False,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
	}

	schinet['v1_l_st_' + p + '_' + lr + '_' + d] = {
		# kinetics400 |  |
	    **data_defaults[d],
		'frame_resize': 112, 'frame_resample': 1, 'input_size': 112,
		'frames_per_clip': 40, 'space_between_frames': 2,
		'eval_frames_per_clip': 40, 'eval_space_between_frames': 2,
		'clip_len': 32, 'clip_location': 'random', 'clip_step': 1,
		'eval_clip_len': 32, 'eval_clip_location': 'center', 'eval_clip_step': 1,
	    'model': 'models.SChiNet.SChiNet',
		'enc_kernel_sizes': ((11, 7, 7), (3, 35, 35)), 'enc_strides': ((4, 2, 2), (2, 16, 16)), 'enc_channels': (64, 64),
	    'chi_stages': ((64, 64, 4, 2), (64, 64, 4, 4), (64, 64, 8, 4), (64, 64, 8, 8)), 'fmap_size': (3, 3), 'head_regroup_post': False,
		'token_dim': (2, 3, 4), 'x_dim': (3, 2), 'res_kernel_size': 3, 'shared_map': False, 'fullconv': False,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
	    'precision': precisions[p], 'qat': p == 'f16-qat', 'stretch': (.1, .1, .1) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		'epochs': 150 if d.startswith('kinetics') else 50, 'sched_milestones': range(50, 150, 10) if d.startswith('kinetics') else range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 25, 'warmup_gamma': 10,
	}


datasets = {'ucf101': 'dataloaders.videodataset.UCF101DataManager',
            'hmdb51': 'dataloaders.videodataset.HMDB51DataManager',
            'kinetics400': 'dataloaders.videodataset.KineticsDataManager',
			'ucf101clp': 'dataloaders.videodataset.UCF101DataManager',
            'hmdb51clp': 'dataloaders.videodataset.HMDB51DataManager',
            'kinetics400clp': 'dataloaders.videodataset.KineticsClipDataManager',
            }

precisions = {
	'f16': {'precision': 'float16', 'qat': False, 'stretch': (1e2, 1e-1, 1e2)},
	'f16-qsqr': {'precision': 'float16', 'qat': False, 'qpow': 2, 'qaff': -0.5, 'q_grad_diff': True, 'stretch': (1e0, 1e-1, 1e0)}, # Reduce range, increase precision of medium values
	'f16-qroot': {'precision': 'float16', 'qat': False, 'qpow': 0.5, 'qaff': -0.25, 'q_grad_diff': False, 'stretch': (1e2, 1e-1, 1e2)}, # Reduce precision of large values, increase precision of small values
	'f16-qat': {'precision': 'float16', 'qat': True, 'stretch': (1, 1, 1)},
	'f32': {'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1)},
}

lrs = {'lr-1e-2': 1e-2, 'lr-5e-3': 5e-3, 'lr-1e-3': 1e-3}
sched_params = {
	'ucf101': {
		'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 10, 'warmup_gamma': 10
	},
    'hmdb51': {
		'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 10, 'warmup_gamma': 10
	},
    'kinetics400': {
		'epochs': 120, 'sched_milestones': range(70, 120, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 20, 'warmup_gamma': 10
	},
	'ucf101clp': {
		'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 10, 'warmup_gamma': 10
	},
    'hmdb51clp': {
		'epochs': 50, 'sched_milestones': range(25, 50, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 10, 'warmup_gamma': 10
	},
    'kinetics400clp': {
		'epochs': 60, 'sched_milestones': range(30, 60, 3), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 5, 'warmup_gamma': 10
	},
}

data_defaults = {d: {
	'data_manager': datasets[d], 'cliplvl_split': d.endswith('clp'), 'clips_per_video': 3 if d.endswith('clp') else 1,
	'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
	'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
	'frame_resize': 112, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
	'frames_per_clip': 20, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
	'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
	'eval_frames_per_clip': 20, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
	'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
	'input_size': 112, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'crops_per_video': 1, #'clips_per_video': 1,
	'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
	'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
	'eval_clip_len': 16, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
	'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
} for d in datasets}
data_defaults_large_clip = {d: {**data_defaults[d], 'frames_per_clip': 80, 'eval_frames_per_clip': 80, 'clip_len': 64, 'eval_clip_len': 64} for d in datasets}


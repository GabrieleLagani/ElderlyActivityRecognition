# 33.23M params, 4169MB, 3m 06s epoch, 93% ucf
config_base = {
	'data_manager': 'dataloaders.videodataset.UCF101DataManager',
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
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2),
	'precision': 'float16', 'qat': False, 'stretch': (100, 100, 100),
	'batch_size': 20, 'lr': 5e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 33.23M params, 10017MB, 6m 15s epoch, 94% ucf
config_base_f32 = {
	'data_manager': 'dataloaders.videodataset.UCF101DataManager',
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
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2),
	'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
	'batch_size': 20, 'lr': 5e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

config_base_qat = {
	'data_manager': 'dataloaders.videodataset.UCF101DataManager',
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
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2),
	'precision': 'float16', 'qat': True, 'stretch': (1, 1, 1),
	'batch_size': 20, 'lr': 5e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 33.23M params, 10417MB, 11m 16s epoch, 94% ucf
config_base_large_clip = {
	'data_manager': 'dataloaders.videodataset.UCF101DataManager',
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
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2),
	'precision': 'float16', 'qat': False, 'stretch': (100, 100, 100),
	'batch_size': 20, 'lr': 5e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}
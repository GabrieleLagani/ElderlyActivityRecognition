datasets = {'ucf101': 'dataloaders.videodataset.UCF101DataManager',
            'hmdb51': 'dataloaders.videodataset.HMDB51DataManager',
            'kinetics400': 'dataloaders.videodataset.KineticsDataManager'}

precisions = {'f16': 'float16', 'f16-qat': 'float16', 'f32': 'float32'}
lrs = {'lr-1e-2': 1e-2, 'lr-5e-3': 5e-3, 'lr-1e-3': 1e-3}

data_defaults = {d: {
	'data_manager': datasets[d],
	'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
	'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
	'frame_resize': 112, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
	'frames_per_clip': 20, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
	'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
	'eval_frames_per_clip': 20, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
	'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
	'input_size': 112, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 1, 'crops_per_video': 1,
	'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
	'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
	'eval_clip_len': 16, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
	'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
} for d in datasets}
data_defaults_large_clip = {d: {**data_defaults, 'frames_per_clip': 80, 'eval_frames_per_clip': 80, 'clip_len': 64, 'eval_clip_len': 64} for d in datasets}


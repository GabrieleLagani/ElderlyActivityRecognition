datasets = {'ucf101': 'dataloaders.videodataset.UCF101DataManager',
            'hmdb51': 'dataloaders.videodataset.HMDB51DataManager', 
            'kinetics400': 'dataloaders.videodataset.KineticsDataManager'}

precisions = {'f16': 'float16', 'f16_qat': 'float16', 'f32': 'float32'}

swin3d = {}
swin3d_large_clip = {}
uniformer = {}
uniformer_large_clip = {}
tubevit = {}
tubevit_large_clip = {}
vivit = {}
vivit_large_clip = {}
timesformer = {}
stam = {}
r3d = {}
r2plus1d = {}
i3d = {}
s3d = {}
x3d = {}
slowfast = {}
slowfast_large_clip = {}


for d, p in [(d, p) for d in datasets for p in precisions]:
	
	# 27.66M params | f16 1703MB, 1m 58s epoch, 81% ucf | f32 3019MB, 2m 00s epoch, 77% ucf
	swin3d[p + '_' + d] = {
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
	    'model': 'models.SwinTransformer3D.SwinTransformer3D',
	    'layer_sizes': (2, 3, 2, 6, 6, 12, 2, 24), 'embed_dim': 32,
		'ff_mult': 4, 'drop': 0, 'drop_path': 0, 'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 4, 4), 'window_size': (2, 7, 7),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 27.66M params | f16 5659MB, 3m 39s epoch, 82% ucf
	swin3d_large_clip[p + '_' + d] = {
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
	    'model': 'models.SwinTransformer3D.SwinTransformer3D',
	    'layer_sizes': (2, 3, 2, 6, 6, 12, 2, 24), 'embed_dim': 32,
		'ff_mult': 4, 'drop': 0, 'drop_path': 0, 'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 4, 4), 'window_size': (2, 7, 7),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 54.80M params | f16 3379MB, 4m 33s epoch, 58% ucf | f32 5793MB, 4m 43s epoch, 66% ucf
	uniformer[p + '_' + d] = {
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
		'model': 'models.UniFormer.UniFormer',
	    'layer_sizes': (3, 4, 8, 3),
		'ff_mult': 4, 'drop': 0, 'norm': 'models.UniFormer.LayerNorm',
		'patch_size': 4, 'window_size': 5,
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90], # 66% ucf
	}

	# 54.80M params | f16 11803MB, 17m 21s epoch, 56% ucf
	uniformer_large_clip[p + '_' + d] = {
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
	    'model': 'models.UniFormer.UniFormer',
	    'layer_sizes': (3, 4, 8, 3),
		'ff_mult': 4, 'drop': 0, 'norm': 'models.UniFormer.LayerNorm',
		'patch_size': 4, 'window_size': 5,
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 16, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}
	
	# 85.66M params | f16 3101MB, 2m 38s epoch, 82% ucf | f32 5021MB, 5m 22s epoch, 85% ucf, 22.1% kinetics
	tubevit[p + '_' + d] = {
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
	    'model': 'models.TubeViT.TubeViT',
	    'layer_sizes': (12, 12), 'embed_dim': 768,
		'ff_mult': 4, 'drop': 0, 'norm': 'torch.nn.LayerNorm',
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}
	
	# 86.57M params | f16 3557MB, 3m 21s epoch, 57% ucf
	tubevit_large_clip[p + '_' + d] = {
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
	    'model': 'models.TubeViT.TubeViT',
	    'layer_sizes': (12, 12), 'embed_dim': 768,
		'ff_mult': 4, 'drop': 0, 'norm': 'torch.nn.LayerNorm',
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}
	
	# 130.13M params | f16 2919 MB, 2m 02s epoch, 79% ucf | f32 4733 MB, 3m 39s epoch, 83% ucf
	vivit[p + '_' + d] = {
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
	    'model': 'models.ViViT.ViViT', 'model_version': 2,
	    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0,  'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 16, 16),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}
	
	# 130.59M params | f16 6925MB, 5m 25s epoch, 82% ucf
	vivit_large_clip[p + '_' + d] = {
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
	    'model': 'models.ViViT.ViViT', 'model_version': 2,
	    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0,  'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 16, 16),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}
	
	# 121.23M params | f16 5713MB, 4m 17s epoch, 84% ucf | f32 10619MB, 11m 17s epoch, 81% ucf
	timesformer[p + '_' + d] = {
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
	    'model': 'models.TimeSFormer.TimeSFormer',
	    'layer_sizes': (12, 12), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm',
		'patch_size': (1, 16, 16),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 118.86M params | f16 6509MB, 4m 22s epoch, 73% ucf | f32 11759MB, 12m 17s epoch, 85% ucf, 29.71% kinetics
	stam[p + '_' + d] = {
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
	    'model': 'models.STAM.STAM',
	    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm',
		'patch_size': (1, 16, 16),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 33.23M params | f16 5123MB, 3m 45s epoch, 88% ucf, 47.49% kinetics | f32 7471MB, 6m 13s epoch, 90% ucf
	r3d[p + '_' + d] = {
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
		'model': 'models.R3D.R3D',
		'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4 if d.startswith('kinetics') else 5e-3,
		'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 33.24M params | f16 ????MB, ???? epoch, ??% ucf | f32 9713MB, 24m 28s epoch, 88% ucf
	r2plus1d[p + '_' + d] = {
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
		'model': 'models.R2Plus1D.R2Plus1D',
		'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
		'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4 if d.startswith('kinetics') else 5e-3,
		'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 12.39M params | f16 ????MB, ???? epoch, 84% ucf, | f32 4353MB, 2m 02s epoch, 90% ucf
	i3d[p + '_' + d] = {
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
	    'model': 'models.I3D.I3D',
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 8.01M params | f16 3179MB, ??? epoch, ??% ucf | f32 5115MB, 2m 0s epoch, 88% ucf
	s3d[p + '_' + d] = {
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
	    'model': 'models.S3D.S3D',
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 3.18M params | f16 3563MB, 2m 21s epoch 90% ucf | f32 2575MB, 2m 30s epoch, 84% ucf
	x3d[p + '_' + d] = {
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
	    'model': 'models.X3D.X3D', 'model_version': 'S',
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4 if d.startswith('kinetics') else 5e-3,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 14.63M params | f16 1811MB, 1m 37s epoch, 90% ucf, 40% kinetics | f32 2235MB, 1m 54s epoch, 92% ucf
	slowfast[p + '_' + d] = {
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
	    'model': 'models.SlowFast.SlowFast',
		'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4 if d.startswith('kinetics') else 5e-3,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}

	# 14.63M params | f16 3337MB, 3m 26s epoch, 91% ucf
	slowfast_large_clip[p + '_' + d] = {
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
	    'model': 'models.SlowFast.SlowFast',
		'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
		'precision': precisions[p], 'qat': p == 'f16_qat', 'stretch': (100, 100, 100) if p == 'f16' else (1, 1, 1),
	    'batch_size': 20, 'lr': 1e-3 if d.startswith('kinetics') else 1e-2, 'wdecay': 5e-4 if d.startswith('kinetics') else 5e-3,
	    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
		#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
	}


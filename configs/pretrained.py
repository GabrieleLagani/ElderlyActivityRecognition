schinet = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': 112, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 80, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*32, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 112, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 64, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*32, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.SChiNet.SChiNet', 'pretrain_path': 'resources/pretrained/schinet.pt',
		'enc_kernel_sizes': ((21, 7, 7), (1, 35, 35)), 'enc_strides': ((8, 2, 2), (1, 16, 16)), 'enc_channels': (64, 64),
		'token_dim': (2, 3, 4), 'x_dim': ('s', 't'), 'alternate_attn': False, 'convattn': True,
		'chi_stages': ((64, 64, 4, 4), (64, 64, 4, 8), (64, 64, 8, 16)),
		'fmap_size': (8, 7), 'head_regroup_post': False, 'res_kernel_size': 3, 'shared_map': False, 'fullconv': True,
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'chi_norm': 'models.SChiNet.TokenBatchNorm', # torch.nn.LayerNorm, #
		'init_mode': 'kaiming_normal', 'disable_wd_for_pos_emb': False, 'drop': 0., 'final_drop': 0.5,
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 120, 'sched_milestones': range(70, 120, 5), 'sched_decay': 0.5,
		#'epochs': 100, 'sched_milestones': [40, 70, 90], 'sched_decay': 0.1,
		'warmup_epochs': 20, 'warmup_gamma': 10,
}

ca3d = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': 112, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 112, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.CA3D.CA3D', 'pretrain_path': 'resources/pretrained/ca3d_f16.pt',
        'patch_size': 2, 'fmap_size': (1, 1), 'token_pool': 'avg', 'cape_reg': 0,
	    'downsample': (1, 2, 1, 2, 1, 1), 'layer_sizes': ((64, 4, 2), (64, 8, 2)), 't_aggr': 'pool', 't_att': 'once',
		'conv_type': 'MultiHeadConv3d', 'res_kernel_size': 3, 'conv_order': 'NCANP', 'res_order': 'ANCANCS', 'rec_col': False,
		'attn_type': 'LocalAttention', 'attn_hidden_rank': 64, 'attn_kernel_size': 5, 'att_order': 'NASNP', 'attn_proj': False,
		'proj_map_type': 'MultiHeadLinear', 'proj_kernel_size': 1, 'headwise_map': True, 'shared_map': False, 'shared_qkv': 'FFF',
		'act': 'torch.nn.ReLU', 'norm': 'models.modutils.BatchNorm', 'init_mode': 'kaiming_normal',
		'pos_emb_mode': 'once', 'disable_wd_for_pos_emb': False, 'drop': 0., 'pos_drop': 0., 'final_drop': 0.,
        'precision': 'float16', 'qat': False, 'stretch': (.1, .1, .1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 150, 'sched_milestones': range(50, 150, 10), 'sched_decay': 0.5,
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
		'warmup_epochs': 10, 'warmup_gamma': 10,
}

# eval_frames_per_clip = (clips_per_video + 1) * clip_len / 2

videomae = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 224, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedVideoMAE.PreTrainedVideoMAE',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

swin3d = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 224, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedSwinTransformer3D.PreTrainedSwinTransformer3D',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

vivit = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 32, 'space_between_frames': 4, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*16, 'eval_space_between_frames': 4, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 224, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 32, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*16, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedViViT.PreTrainedViViT',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 4, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

timesformer = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 224, 'da_time_scale_rel_delta': 0.,  'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedTimesFormer.PreTrainedTimesFormer',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

r3d = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 112, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedR3D.PreTrainedR3D',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

r2plus1d = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 112, 'da_time_scale_rel_delta': 0.,  'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedR2Plus1D.PreTrainedR2Plus1D',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

s3d = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 224, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedS3D.PreTrainedS3D',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

x3d = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 16, 'space_between_frames': 5, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*8, 'eval_space_between_frames': 5, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 256, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 16, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*8, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedX3D.PreTrainedX3D',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

movinet_a2 = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 250, 'space_between_frames': 1, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 250, 'eval_space_between_frames': 1, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 224, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 1, 'crops_per_video': 3,
        'clip_len': 250, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 250, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.MoViNet.MoViNet',
		'movinet_model': 'A2', 'movinet_subclip_len': 50, 'movinet_pretrained': True,
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

slowfast = {
        'data_manager': 'dataloaders.videodataset.KineticsDataManager',
        'augment_manager': 'dataloaders.videodataset.LightAugmentManager',
        'num_workers': 4, 'workers_on_gpu': False, 'processing_dtype': 'uint8',
        'frame_resize': None, 'frame_resample': 1, 'min_frame_resample': None, 'max_frame_resample': None, 'auto_resample_num_frames': 80,
        'frames_per_clip': 32, 'space_between_frames': 2, 'min_space_between_frames': None, 'max_space_between_frames': None,
        'frame_jitter': None, 'auto_frames': None, 'min_auto_frames': None, 'max_auto_frames': None,
        'eval_frames_per_clip': 11*16, 'eval_space_between_frames': 2, 'eval_auto_frames': None, 'eval_frame_jitter': None,
		'preproc_frames_per_clip': 20, 'preproc_space_between_frames': 1, 'preproc_auto_frames': None, 'preproc_frame_jitter': None,
        'input_size': 224, 'da_time_scale_rel_delta': 0., 'multicrop_test': False, 'clips_per_video': 10, 'crops_per_video': 3,
        'clip_len': 32, 'clip_location': 'random', 'clip_step': 1, 'min_clip_step': None, 'max_clip_step': None,
        'auto_len': None, 'min_auto_len': None, 'max_auto_len': None, 'data_backend': 'default',
        'eval_clip_len': 11*16, 'eval_clip_location': 'center', 'eval_clip_step': 1, 'eval_auto_len': None,
		'preproc_clip_len': 16, 'preproc_clip_location': 'center', 'preproc_clip_step': 1, 'preproc_auto_len': None,
        'model': 'models.hub.PreTrainedSlowFast.PreTrainedSlowFast',
        'precision': 'float32', 'qat': False, 'stretch': (1, 1, 1),
        'batch_size': 20, 'eval_batch_size': 1, 'lr': 1e-3, 'wdecay': 5e-5,
        'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
        #'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}
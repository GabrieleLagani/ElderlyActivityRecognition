# 33.23M params, 4169MB, 3m 06s epoch, 93% ucf
config_base = {
	'dataset': 'UCF-101', # 'hmdb-51
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2),
	'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
	'batch_size': 20, 'lr': 5e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 33.23M params, 10017MB, 6m 15s epoch, 94% ucf
config_base_f32 = {
	'dataset': 'UCF-101', # 'hmdb-51
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2),
	'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
	'batch_size': 20, 'lr': 5e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 33.23M params, 10417MB, 11m 16s epoch, 94% ucf
config_base_large_clip = {
	'dataset': 'UCF-101', # 'hmdb-51
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2),
	'clip_len': 64,
	'precision': 'float16', 'stretch': (100, 100, 100),
	'batch_size': 20, 'lr': 5e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 27.66M params, 1703MB, 1m 58s epoch, 81% ucf
swin3d_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.SwinTransformer3D.SwinTransformer3D',
    'layer_sizes': (2, 3, 2, 6, 6, 12, 2, 24), 'embed_dim': 32, 'ff_mult': 4,
	'drop': 0, 'drop_path': 0, 'norm': 'torch.nn.LayerNorm',
	'clip_len': 16, 'patch_size': (4, 4, 4), 'window_size': (2, 7, 7),
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 27.66M params, 3019MB, 2m 00s epoch, 77% ucf
swin3d_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.SwinTransformer3D.SwinTransformer3D',
    'layer_sizes': (2, 3, 2, 6, 6, 12, 2, 24), 'embed_dim': 32, 'ff_mult': 4,
	'drop': 0, 'drop_path': 0, 'norm': 'torch.nn.LayerNorm',
	'clip_len': 16, 'patch_size': (4, 4, 4), 'window_size': (2, 7, 7),
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 27.66M params, 5659MB, 3m 39s epoch, 82% ucf
swin3d_large_clip_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.SwinTransformer3D.SwinTransformer3D',
    'layer_sizes': (2, 3, 2, 6, 6, 12, 2, 24), 'embed_dim': 32, 'ff_mult': 4,
	'drop': 0, 'drop_path': 0, 'norm': 'torch.nn.LayerNorm',
	'clip_len': 64, 'patch_size': (4, 4, 4), 'window_size': (2, 7, 7),
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 54.80M params, 3379MB, 4m 33s epoch, 58% ucf
uniformer_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.UniFormer.UniFormer',
    'layer_sizes': (3, 4, 8, 3), 'ff_mult': 4, 'drop': 0, 'norm': 'models.UniFormer.LayerNorm',
	'clip_len': 16, 'patch_size': 4, 'window_size': 5,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90], # 66% ucf
}

#54.80M params, 5793MB, 4m 43s epoch, 66% ucf
uniformer_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.UniFormer.UniFormer',
    'layer_sizes': (3, 4, 8, 3), 'ff_mult': 4, 'drop': 0, 'norm': 'models.UniFormer.LayerNorm',
	'clip_len': 16, 'patch_size': 4, 'window_size': 5,
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90], # 66% ucf
}

# 54.80M params, 11803MB, 17m 21s epoch, 56% ucf
uniformer_large_clip_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.UniFormer.UniFormer',
    'layer_sizes': (3, 4, 8, 3), 'ff_mult': 4, 'drop': 0, 'norm': 'models.UniFormer.LayerNorm',
	'clip_len': 64, 'patch_size': 4, 'window_size': 5,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 16, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 865.66M params, 3101MB, 2m 38s epoch, 82% ucf
tubevit_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.TubeViT.TubeViT',
    'layer_sizes': (12, 12), 'embed_dim': 768, 'ff_mult': 4, 'norm': 'torch.nn.LayerNorm',
	'drop': 0, 'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 85.66M params, 5021MB, 5m 22s epoch, 85% ucf
tubevit_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.TubeViT.TubeViT',
    'layer_sizes': (12, 12), 'embed_dim': 768, 'ff_mult': 4, 'norm': 'torch.nn.LayerNorm',
	'drop': 0, 'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 86.57M params, 3557MB, 3m 21s epoch, 57% ucf
tubevit_large_clip_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.TubeViT.TubeViT',
    'layer_sizes': (12, 12), 'embed_dim': 768, 'ff_mult': 4, 'norm': 'torch.nn.LayerNorm',
	'drop': 0, 'clip_len': 64,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

#130.13M params, 2919 MB, 2m 02s epoch, 79% ucf
vivit_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.ViViT.ViViT', 'model_version': 2,
    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'ff_mult': 4,
	'drop': 0,  'norm': 'torch.nn.LayerNorm', 'token_pool': 'first',
	'clip_len': 16, 'patch_size': (4, 16, 16),
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

#130.13M params, 4733 MB, 3m 39s epoch, 83% ucf
vivit_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.ViViT.ViViT', 'model_version': 2,
    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'ff_mult': 4,
	'drop': 0,  'norm': 'torch.nn.LayerNorm', 'token_pool': 'first',
	'clip_len': 16, 'patch_size': (4, 16, 16),
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 130.59M params, 6925MB, 5m 25s epoch, 82% ucf
vivit_large_clip_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.ViViT.ViViT', 'model_version': 2,
    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'ff_mult': 4,
	'drop': 0,  'norm': 'torch.nn.LayerNorm', 'token_pool': 'first',
	'clip_len': 64, 'patch_size': (4, 16, 16),
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 121.23M params, 5713MB, 4m 17s epoch, 84% ucf
timesformer_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.TimeSFormer.TimeSFormer',
    'layer_sizes': (12, 12), 'embed_dim': 768, 'ff_mult': 4,
	'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm', 'token_pool': 'first',
	'clip_len': 16, 'patch_size': (1, 16, 16),
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 121.23M params, 10619MB, 11m 17s epoch, 81% ucf
timesformer_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.TimeSFormer.TimeSFormer',
    'layer_sizes': (12, 12), 'embed_dim': 768, 'ff_mult': 4,
	'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm', 'token_pool': 'first',
	'clip_len': 16, 'patch_size': (1, 16, 16),
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 118.86M params, 6509MB, 4m 22s epoch, 73% ucf
stam_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.STAM.STAM',
    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'ff_mult': 4,
	'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm', 'token_pool': 'first',
	'clip_len': 16, 'patch_size': (1, 16, 16),
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

#118.86M params, 11759MB, 12m 17s epoch, 85% ucf
stam_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.STAM.STAM',
    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'ff_mult': 4,
	'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm', 'token_pool': 'first',
	'clip_len': 16, 'patch_size': (1, 16, 16),
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-4, 'disable_wd_for_pos_emb': True,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 33.23M params, 5123MB, 3m 45s epoch, 88% ucf
r3d_ucf = {
	'dataset': 'UCF-101', # 'hmdb-51
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
	'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
	'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 33.24M params, 7471MB, 6m 13s epoch, 90% ucf
r3d_f32_ucf = {
	'dataset': 'UCF-101', # 'hmdb-51
	'model': 'models.R3D.R3D',
	'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
	'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
	'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}


r2plus1d_ucf = {
	'dataset': 'UCF-101', # 'hmdb-51
	'model': 'models.R2Plus1D.R2Plus1D',
	'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
	'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
	'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 33.24M params, 9713MB, 24m 28s epoch, 88% ucf
r2plus1d_f32_ucf = {
	'dataset': 'UCF-101', # 'hmdb-51
	'model': 'models.R2Plus1D.R2Plus1D',
	'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
	'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
	'batch_size': 12, 'lr': 1e-2, 'wdecay': 5e-3,
	'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 12.39M params, ????MB, ???? epoch, 84% ucf
i3d_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.I3D.I3D',
	'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-3,
    #'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

#12.39M params, 4353MB, 2m 02s epoch, 90% ucf
i3d_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.I3D.I3D',
	'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 8.01M params, 3179MB, ??? epoch, ??% ucf
s3d_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.S3D.S3D',
	'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4,
    #'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 8.01M params, 5115MB, 2m 0s epoch, 88% ucf
s3d_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.S3D.S3D',
	'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-4,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 3.18M params, 3563MB, 2m 21s epoch 90% ucf
x3d_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.X3D.X3D', 'model_version': 'S',
	'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-3,
    #'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 3.18M params, 2575MB, 2m 30s epoch, 84% ucf
x3d_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.X3D.X3D', 'model_version': 'S', # XL: 10.48M params, 12015MB, 5m 24s epoch, 82% ucf
	'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-3,
    'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	#'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 14.63M params, 1811MB, 1m 37s epoch, 90% ucf
slowfast_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.SlowFast.SlowFast',
	'clip_len': 16,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-3,
    #'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 14.63M params, 2235MB, 1m 54s epoch, 92% ucf
slowfast_f32_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.SlowFast.SlowFast',
	'clip_len': 16,
	'precision': 'float32', 'stretch': (1, 1, 1),
    'batch_size': 20, 'lr': 1e-3, 'wdecay': 5e-3,
    #'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}

# 14.63M params, 3337MB, 3m 26s epoch, 91% ucf
slowfast_large_clip_ucf = {
    'dataset': 'UCF-101', # 'hmdb-51
    'model': 'models.SlowFast.SlowFast',
	'clip_len': 64,
	'precision': 'float16', 'stretch': (100, 100, 100),
    'batch_size': 20, 'lr': 1e-2, 'wdecay': 5e-3,
    #'epochs': 50, 'sched_decay': 0.5, 'sched_milestones': range(25, 50, 5),
	'epochs': 100, 'sched_decay': 0.1, 'sched_milestones': [40, 70, 90],
}


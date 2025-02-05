from .defaults import *

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
movinet = {}
slowfast = {}
slowfast_large_clip = {}


for d, p, lr in [(d, p, lr) for d in datasets for p in precisions for lr in lrs]:

	swin3d[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.SwinTransformer3D.SwinTransformer3D',
	    'layer_sizes': (2, 3, 2, 6, 6, 12, 2, 24), 'embed_dim': 32,
		'ff_mult': 4, 'drop': 0, 'drop_path': 0, 'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 4, 4), 'window_size': (2, 7, 7),
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	swin3d_large_clip[p + '_' + lr + '_' + d] = {
	    **data_defaults_large_clip[d],
	    'model': 'models.SwinTransformer3D.SwinTransformer3D',
	    'layer_sizes': (2, 3, 2, 6, 6, 12, 2, 24), 'embed_dim': 32,
		'ff_mult': 4, 'drop': 0, 'drop_path': 0, 'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 4, 4), 'window_size': (2, 7, 7),
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	uniformer[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
		'model': 'models.UniFormer.UniFormer',
	    'layer_sizes': (3, 4, 8, 3),
		'ff_mult': 4, 'drop': 0, 'norm': 'models.UniFormer.LayerNorm',
		'patch_size': 4, 'window_size': 5,
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	uniformer_large_clip[p + '_' + lr + '_' + d] = {
	    **data_defaults_large_clip[d],
	    'model': 'models.UniFormer.UniFormer',
	    'layer_sizes': (3, 4, 8, 3),
		'ff_mult': 4, 'drop': 0, 'norm': 'models.UniFormer.LayerNorm',
		'patch_size': 4, 'window_size': 5,
		**precisions[p],
	    'batch_size': 16, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	tubevit[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.TubeViT.TubeViT',
	    'layer_sizes': (12, 12), 'embed_dim': 768,
		'ff_mult': 4, 'drop': 0, 'norm': 'torch.nn.LayerNorm',
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	tubevit_large_clip[p + '_' + lr + '_' + d] = {
	    **data_defaults_large_clip[d],
	    'model': 'models.TubeViT.TubeViT',
	    'layer_sizes': (12, 12), 'embed_dim': 768,
		'ff_mult': 4, 'drop': 0, 'norm': 'torch.nn.LayerNorm',
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	vivit[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.ViViT.ViViT', 'model_version': 2,
	    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0,  'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 16, 16),
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	vivit_large_clip[p + '_' + lr + '_' + d] = {
	    **data_defaults_large_clip[d],
	    'model': 'models.ViViT.ViViT', 'model_version': 2,
	    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0,  'norm': 'torch.nn.LayerNorm',
		'patch_size': (4, 16, 16),
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	timesformer[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.TimeSFormer.TimeSFormer',
	    'layer_sizes': (12, 12), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm',
		'patch_size': (1, 16, 16),
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	stam[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.STAM.STAM',
	    'layer_sizes': (12, 12, 6, 8), 'embed_dim': 768, 'token_pool': 'first',
		'ff_mult': 4, 'drop': 0, 'drop_path': 0., 'norm': 'torch.nn.LayerNorm',
		'patch_size': (1, 16, 16),
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4, 'disable_wd_for_pos_emb': True,
	    **sched_params[d],
	}

	r3d[p + '_' + lr + '_' + d] = {
		**data_defaults[d],
		'model': 'models.R3D.R3D',
		'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
		**precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	r2plus1d[p + '_' + lr + '_' + d] = {
		**data_defaults[d],
		'model': 'models.R2Plus1D.R2Plus1D',
		'layer_sizes': (2, 2, 2, 2), #(2, 2, 4, 8),
		**precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	i3d[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.I3D.I3D',
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4,
	    **sched_params[d],
	}

	s3d[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.S3D.S3D',
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-4,
	    **sched_params[d],
	}

	x3d[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.X3D.X3D', 'model_version': 'XL',
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
	    **sched_params[d],
	}

	movinet[p + '_' + lr + '_' + d] = {
		**data_defaults[d],
		'model': 'models.MoViNet.MoViNet',
		'movinet_model': 'A2',
		**precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
		**sched_params[d],
	}

	slowfast[p + '_' + lr + '_' + d] = {
	    **data_defaults[d],
	    'model': 'models.SlowFast.SlowFast',
		'layer_sizes': (2, 2, 4, 8),
		**precisions[p],
	    'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
	    **sched_params[d],
	}

	slowfast_large_clip[p + '_' + lr + '_' + d] = {
	    **data_defaults_large_clip[d],
	    'model': 'models.SlowFast.SlowFast',
		'layer_sizes': (2, 2, 4, 8),
		**precisions[p],
		'batch_size': 20, 'lr': lrs[lr], 'wdecay': 5e-5 if d.startswith('kinetics') else 5e-3,
	    **sched_params[d],
	}


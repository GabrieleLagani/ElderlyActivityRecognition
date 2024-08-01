datasets = ['ucf101', 'hmdb51', 'kinetics400']
all = {d: [] for d in datasets}

for d in datasets:
	all[d] += [{'config': 'configs.baselines.swin3d[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.swin3d[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.swin3d[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.swin3d[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.swin3d[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.swin3d[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.uniformer[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.uniformer[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.uniformer[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.uniformer[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.uniformer[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.uniformer[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.tubevit[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.tubevit[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.tubevit[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.tubevit[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.tubevit[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--
	all[d] += [{'config': 'configs.baselines.tubevit[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]

	all[d] += [{'config': 'configs.baselines.vivit[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.vivit[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.vivit[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.vivit[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.vivit[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.vivit[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.timesformer[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.timesformer[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.timesformer[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.timesformer[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.timesformer[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.timesformer[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.stam[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.stam[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.stam[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.stam[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.stam[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.stam[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.r3d[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r3d[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r3d[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r3d[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r3d[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r3d[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.r2plus1d[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r2plus1d[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r2plus1d[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r2plus1d[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r2plus1d[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.r2plus1d[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.i3d[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.i3d[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.i3d[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.i3d[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.i3d[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.i3d[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--

	all[d] += [{'config': 'configs.baselines.s3d[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.s3d[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.s3d[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.s3d[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.s3d[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--
	all[d] += [{'config': 'configs.baselines.s3d[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]

	all[d] += [{'config': 'configs.baselines.x3d[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.x3d[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.x3d[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.x3d[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.x3d[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--
	all[d] += [{'config': 'configs.baselines.x3d[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]

	all[d] += [{'config': 'configs.baselines.slowfast[f16_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.slowfast[f16_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.slowfast[f16-qat_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.slowfast[f16-qat_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	all[d] += [{'config': 'configs.baselines.slowfast[f32_lr-1e-2_' + d + ']', 'seeds': [0], 'dataseeds': [100]}] # <--
	all[d] += [{'config': 'configs.baselines.slowfast[f32_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]


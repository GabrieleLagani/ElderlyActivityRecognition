datasets = ['ucf101', 'hmdb51', 'kinetics400', 'ucf101clp', 'hmdb51clp', 'kinetics400clp']
precisions = ['f16, f16-qat', 'f16-sqr', 'f16-root', 'f32']
all = {d: [] for d in datasets}

for d in datasets:
	for p in precisions: all[d] += [{'config': 'configs.baselines.swin3d[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.uniformer[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.tubevit[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.vivit[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.timesformer[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.stam[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.r3d[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.r2plus1d[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.i3d[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.s3d[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.x3d[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.movinet[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]
	for p in precisions: all[d] += [{'config': 'configs.baselines.slowfast[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]


datasets = ['ucf101', 'hmdb51', 'kinetics400', 'ucf101clp', 'hmdb51clp', 'kinetics400clp']
precisions = ['f16, f16-qat', 'f16-sqr', 'f16-root', 'f32']
all = {d: [] for d in datasets}

for d in datasets:
	for p in precisions: all[d] += [{'config': 'configs.ca3d.ca3d[' + p + '_lr-1e-3_' + d + ']', 'seeds': [0], 'dataseeds': [100]}]


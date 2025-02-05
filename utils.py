import os
import csv
import math
import random
from importlib import import_module

import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn.modules.utils import _triple


# Set rng seed
def set_rng_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Set rng state
def set_rng_state(state):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.setstate(state['python_rng'])
	np.random.set_state(state['numpy_rng'])
	torch.set_rng_state(state['pytorch_rng'])
	torch.cuda.set_rng_state_all(state['pytorch_rng_cuda'])

# Get rng state
def get_rng_state():
	state = {}
	state['python_rng'] = random.getstate()
	state['numpy_rng'] = np.random.get_state()
	state['pytorch_rng'] = torch.get_rng_state()
	state['pytorch_rng_cuda'] = torch.cuda.get_rng_state_all()
	return state

# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# Transforms shape tuple to size by multiplying the shape values
def shape2size(shape):
	size = 1
	for s in shape: size *= s
	return size

# Returns padding size corresponding to padding "same" for a given kernel size
def get_padding_same(kernel_size):
	kernel_size = _triple(kernel_size)
	return [(k - 1) // 2 for k in kernel_size]

# Returns output size after convolution
def get_conv_output_size(input_size, kernel_size, stride=1, padding=0):
	if padding == 'same': padding = get_padding_same(kernel_size)[0]
	return ((input_size + 2*padding - kernel_size) // stride) + 1

# Save data to csv file
def update_csv(results, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, mode='w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for name, entries in results.items():
			writer.writerow([name + '_epoch'] + list(entries.keys()))
			writer.writerow([name] + list(entries.values()))

# Add an entry containing the seed of a training iteration and the test accuracy of the corresponding model to a csv file
def update_iter_csv(iter_id, result, path, ci_levels=(0.9, 0.95, 0.98, 0.99, 0.995)):
	AVG_KEY = 'AVG'
	CI_KEYS = {ci_lvl: str(ci_lvl*100) + "% CI" for ci_lvl in ci_levels}
	HEADER = ('ITER_ID', 'RESULT')
	os.makedirs(os.path.dirname(path), exist_ok=True)
	d = {}
	try:
		with open(path, 'r') as csv_file:
			reader = csv.reader(csv_file)
			d = dict(reader)
			d.pop(HEADER[0], None)
			d.pop(AVG_KEY, None)
			for ci_lvl in ci_levels: d.pop(CI_KEYS[ci_lvl], None)
	except: pass
	d[str(iter_id)] = str(result)
	with open(path, mode='w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(HEADER)
		for k, v in d.items(): writer.writerow([k, v])
		if len(d) > 1:
			values = list(map(float, d.values()))
			avg = sum(values)/len(values)
			se = st.sem(values)
			writer.writerow([AVG_KEY, str(avg)])
			for ci_lvl in ci_levels:
				ci = st.t.interval(ci_lvl, len(values) - 1, loc=avg, scale=se)
				ci_str = "+/- " + str((ci[1] - ci[0])/2)
				writer.writerow([CI_KEYS[ci_lvl], ci_str])

# Save a figure showing train and validation results in the specified file
def save_trn_curve_plot(train_result_data, val_result_data, path, label='result'):
	graph = plt.axes(xlabel='epoch', ylabel=label)
	graph.plot(list(train_result_data.keys()), list(train_result_data.values()), label='train')
	graph.plot(list(val_result_data.keys()), list(val_result_data.values()), label='val.')
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig = graph.get_figure()
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Save state dictionary file to specified path
def save_dict(state_dict, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state_dict, path)
	
# Load state dictionary file from specified path
def load_dict(path, device='cpu'):
	return torch.load(path, map_location=device)

# Remaps tensors in state dict to the desired dtype
def map_dtype(state_dict, dtype=torch.float):
	return {k: map_dtype(v, dtype) if isinstance(v, dict) else (v.to(dtype=dtype) if isinstance(v, torch.Tensor) else v) for k, v in state_dict.items()}

# Retrieve a custom module or object provided by the user by full name in dot notation as string. If the object is a
# dictionary, it is possible to retrieve a specific element of the dictionary with the square bracket indexing notation.
# NB: dictionary keys must always be strings.
def retrieve(name):
	if name is None: return None
	
	if '[' in name:
		name, key = name.split('[', 1)
		key = key.rsplit(']', 1)[0]
		prefix, suffix = name.rsplit('.', 1)
		return getattr(import_module(prefix), suffix)[key]
	
	prefix, suffix = name.rsplit('.', 1)
	return getattr(import_module(prefix), suffix)

def trunc_normal_(tensor, mean=0., std=1., a=-2, b=2):
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly initialized tensor with values from [2l-1, 2u-1]
		values = 2 * (torch.rand(tensor.shape, dtype=torch.float32) * (u - l) + l) - 1 # Explicitly use float32 as dtype, because erfinv_ function dowsn't work with other dtypes

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		values.erfinv_()

		# Transform to proper mean, std
		values.mul_(std * math.sqrt(2.))
		values.add_(mean)

		# Clamp to ensure it's in the proper range
		values.clamp_(min=a, max=b)

		# Fill tensor with values and return it
		tensor[:] = values.to(dtype=tensor.dtype)
		return tensor

# Neural network weight initialization
def init_model_params(model, mode='kaiming_normal'):
	for m in model.modules():
		if isinstance(m, (nn.Conv3d, nn.Linear)):
			if mode == 'kaiming_normal':
				nn.init.kaiming_normal_(m.weight)
			elif mode == 'xavier_normal':
				nn.init.xavier_normal_(m.weight)
			elif mode == 'trunc_normal':
				nn.init.trunc_normal_(m.weight, std=.02)
			elif mode == 'kaiming_trunc_normal':
				n = shape2size(m.weight.shape) // m.weight.shape[0]
				trunc_normal_(m.weight, std=(2. / n)**0.5)
			else:
				raise NotImplementedError("Init mode {} not available".format(mode))
		if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm)):
			nn.init.ones_(m.weight)
			nn.init.zeros_(m.bias)

# Count neural network parameters
def count_params(model):
	return sum(p.numel() for n, p in model.named_parameters() if not n.startswith('pre_params'))

# Set weight decay to zero for positional embedding model params
def disable_wd_for_pos_emb(model):
	def_params = {'params': [], 'param_names': []}
	no_wd_params = {'params': [], 'param_names': [], 'weight_decay': 0}
	for n, p in model.named_parameters():
		if not n.startswith('pre_params'):
			if hasattr(model, 'disable_wd_for_pos_emb') and model.disable_wd_for_pos_emb and (n.endswith('cls_token') or n.endswith('pos_embed')):
				no_wd_params['params'].append(p)
				no_wd_params['param_names'].append(n)
			else:
				def_params['params'].append(p)
				def_params['param_names'].append(n)
	return [def_params, no_wd_params]

# Computes the hidden feature map shape of a model by feeding it a fake input. The model must implement the feature
# mapping in a forward_features method. This method is useful both to obtain the shape of feature maps, in order to
# initialize successive layers (e.g. a final linear classifier), and also to initialize layers which perform lazy
# initialization.
def get_fmap_shape(model):
	with torch.no_grad():
		model_state = model.training
		model.eval()
		x = torch.ones(1, *model.get_default_input_shape()) # Fake input
		x = model.forward_features(x)
		model.train(model_state)
		return x.shape[1:]



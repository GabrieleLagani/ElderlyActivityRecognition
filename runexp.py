import argparse
import os
from tqdm import tqdm
import timeit
import math
import copy

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import params as P
import utils


# TODO: Models to torch hub
# TODO: Add optical flow preprocessing


class WarmUpMultiStepLR:
	def __init__(self, warmup_epochs, warmup_gamma, milestones, gamma):
		self.warmup_epochs = warmup_epochs
		self.warmup_gamma = warmup_gamma
		self.milestones = milestones
		self.gamma = gamma

	def __call__(self, epoch):
		k = sum([1 for m in self.milestones if m <= epoch])
		return ((1 + (self.warmup_gamma - 1) * epoch / self.warmup_epochs) if epoch < self.warmup_epochs else self.warmup_gamma) * (self.gamma**k)

class AggregateEvaluator:
	def __init__(self, num_items, num_classes, crops_per_item, criterion):
		self.num_items = num_items
		self.num_classes = num_classes
		self.crops_per_item = crops_per_item
		self.criterion = criterion
		self.reset()

	def reset(self):
		self.outputs = torch.zeros([self.num_items, self.num_classes], dtype=P.DTYPE, device=P.DEVICE)
		self.outputs_smax = torch.zeros([self.num_items, self.num_classes], dtype=P.DTYPE, device=P.DEVICE)
		self.labels = torch.zeros([self.num_items], dtype=torch.int64, device=P.DEVICE)
		self.idx2row = {}

	def update(self, idx, outputs, labels):
		with torch.no_grad():
			idx = idx // self.crops_per_item
			rows = []
			for i in idx:
				if i not in self.idx2row: self.idx2row[i] = len(self.idx2row)
				rows.append(self.idx2row[i])
			rows = torch.tensor(rows, dtype=torch.int64, device=P.DEVICE)
			self.outputs[rows] += outputs
			self.outputs_smax[rows] += outputs.softmax(dim=1)
			self.labels[rows] = labels

	def compute(self):
		with torch.no_grad():
			outputs = self.outputs
			outputs_smax = self.outputs_smax
			labels = self.labels
			outputs = outputs / self.crops_per_item
			loss = self.criterion(outputs, labels).item()
			#outputs = outputs_smax
			hits = (labels == torch.max(outputs, dim=1)[1]).int().sum().item()
			hits_t5 = (labels.unsqueeze(-1) == outputs.topk(5, 1)[1]).int().sum().item()
			count = len(labels)
			return loss, hits/count, hits_t5/count

class Quantizer:
	def __init__(self, cfg):

		self.debug = True

		self.config = cfg
		self.stretch_h, self.stretch_v, self.stretch_g = self.config.get('stretch', (1, 1, 1))
		self.precision = self.config.get('precision', 'float32')
		if self.precision != 'float32' and P.DEVICE == 'cpu':
			raise RuntimeError("Only float32 precision is supported when using cpu device")
		if self.precision == 'float16': P.DTYPE = torch.float16
		if self.precision == 'float64': P.DTYPE = torch.float64
		self.qat = self.config.get('qat', False)
		self.qpow = self.config.get('qpow', 1)
		self.qaff = self.config.get('qaff', 0)
		self.grad_diff = self.config.get('q_grad_diff', False)
		self.preparam_precision = torch.float32 if self.qat else P.DTYPE

	def _invalid_value(self, x):
		return self.debug and x is not None and ( torch.any(x.isnan()) ) #or torch.any(x.isinf()) )

	def _signed_pow(self, x, p):
		return x.sign() * x.abs().pow(p)

	def _d_signed_pow(self, x, p):
		return p * x.abs().pow(p-1)

	def _clamp(self, x, min=None, max=None):
		x = x.clone()
		if min is not None: x[x < min] = min
		if max is not None: x[x > max] = max
		return x

	def _d_clamp(self, x, y, min=None, max=None):
		x = x.clone()
		if min is not None: x[y < min] = 0
		if max is not None: x[y > max] = 0
		return x

	def pre_params_enabled(self):
		return self.stretch_h != 1 or self.qat or self.qpow != 1

	@torch.no_grad()
	def param_to_pre_param(self, p):
		out1 = ( self._signed_pow(self.stretch_h * (p - self.qaff).to(dtype=self.preparam_precision), self.qpow) if self.qpow >= 1 else
				( (self.stretch_h**self.qpow) * self._signed_pow((p - self.qaff).to(dtype=self.preparam_precision), self.qpow) ) )
		out1 = out1 - self._signed_pow(self.stretch_h * (torch.zeros_like(p) - self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out2 = ( self._signed_pow(self.stretch_h * (p + self.qaff).to(dtype=self.preparam_precision), self.qpow) if self.qpow >= 1 else
				( (self.stretch_h**self.qpow) * self._signed_pow((p + self.qaff).to(dtype=self.preparam_precision), self.qpow) ) )
		out2 = out2 - self._signed_pow(self.stretch_h * (torch.zeros_like(p) + self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out = self._clamp(out1, min=0) + self._clamp(out2, max=0)
		return out

	@torch.no_grad()
	def _param_to_preparam_grad_diff(self, p):
		if p.grad is None: return None
		pre_p = self.param_to_pre_param(p)
		if self._invalid_value(pre_p): raise RuntimeError("Pre-parameter is nan")

		offs1 = self._signed_pow(self.stretch_g * (torch.zeros_like(p) - self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out1 = self._d_signed_pow(pre_p + offs1, 1/self.qpow) / self.stretch_g
		out1 = out1 #+ self.qaff
		offs2 = self._signed_pow(self.stretch_g * (torch.zeros_like(p) + self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out2 = self._d_signed_pow(pre_p + offs2, 1/self.qpow) / self.stretch_g
		out2 = out2 #- self.qaff
		out = self._d_clamp(out1, pre_p, min=0) + self._d_clamp(out2, pre_p, max=0)
		return (p.grad.to(dtype=self.preparam_precision) / self.stretch_v) * out

	@torch.no_grad()
	def _param_to_preparam_grad(self, p):
		if p.grad is None: return None
		out1 = ( self._signed_pow((self.stretch_g/self.stretch_v) * (p.grad - self.qaff).to(dtype=self.preparam_precision), self.qpow) if self.qpow >= 1 else
				( ((self.stretch_g/self.stretch_v)**self.qpow) * self._signed_pow((p.grad - self.qaff).to(dtype=self.preparam_precision), self.qpow) ) )
		out1 = out1 - self._signed_pow(self.stretch_h * (torch.zeros_like(p) - self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out2 = ( self._signed_pow((self.stretch_g/self.stretch_v) * (p.grad + self.qaff).to(dtype=self.preparam_precision), self.qpow) if self.qpow >= 1 else
				( ((self.stretch_g/self.stretch_v)**self.qpow) * self._signed_pow((p.grad + self.qaff).to(dtype=self.preparam_precision), self.qpow) ) )
		out2 = out2 - self._signed_pow(self.stretch_h * (torch.zeros_like(p) + self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out = self._clamp(out1, min=0) + self._clamp(out2, max=0)
		return out

	@torch.no_grad()
	def param_to_pre_param_grad(self, p):
		if self._invalid_value(p.grad): raise RuntimeError("Parameter gradient is nan")
		out = self._param_to_preparam_grad_diff(p) if self.grad_diff else self._param_to_preparam_grad(p)
		if self._invalid_value(out): raise RuntimeError("Pre-parameter gradient is nan.")
		return out

	@torch.no_grad()
	def pre_param_to_param(self, p):
		if self._invalid_value(p): raise RuntimeError("Pre-parameter is nan")
		offs1 = self._signed_pow(self.stretch_h * (torch.zeros_like(p) - self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out1 = ((self._signed_pow((p + offs1), 1 / self.qpow) / self.stretch_h) if self.qpow >= 1 else
				self._signed_pow((p + offs1) / (self.stretch_h ** (self.qpow)), 1 / self.qpow))
		# out1 = ( (self._signed_pow((p - offs1), 1/self.qpow) / self.stretch_h) if self.qpow >= 1 else
		#       self._signed_pow((p - offs1) / (self.stretch_h**(self.qpow)), 1/self.qpow) )
		out1 = out1 + self.qaff
		offs2 = self._signed_pow(self.stretch_h * (torch.zeros_like(p) + self.qaff).to(dtype=self.preparam_precision), self.qpow)
		out2 = ((self._signed_pow((p + offs2), 1 / self.qpow) / self.stretch_h) if self.qpow >= 1 else
				self._signed_pow((p + offs2) / (self.stretch_h ** (self.qpow)), 1 / self.qpow))
		out2 = out2 - self.qaff
		# out2 = out2 + self.qaff
		out = self._clamp(out1, min=0) + self._clamp(out2, max=0)
		if self._invalid_value(out): raise RuntimeError("Parameter is nan")
		return out

	def rescale_loss(self, l):
		return self.stretch_v * l


class Experiment:
	def __init__(self, config_name, mode, restart, seed, dataseed, token):
		self.config_name = config_name
		self.mode = mode
		self.seed = seed
		self.dataseed = dataseed
		self.token = token
		print("\n****    Config {} | Mode {} | Iter {}    ****".format(self.config_name, self.mode, self.seed))

		self.config = utils.retrieve(self.config_name)
		print("Experiment configuration: {}".format(self.config))
		print("Dataseed: {}".format(self.dataseed))

		print("\nInitializing experiment...")

		self.iter_dir = os.path.join(P.RESULT_FOLDER, self.config_name.replace('.', os.sep))
		self.result_dir = os.path.join(self.iter_dir, 'iter{}'.format(self.seed))
		self.checkpoint_path = os.path.join(self.result_dir, 'checkpoint.pt')
		self.result_path = os.path.join(self.result_dir, 'results.csv')
		self.test_result_path = os.path.join(self.iter_dir, 'acc')
		self.t5_test_result_path = os.path.join(self.iter_dir, 'acc_t5')
		self.best_model_path = os.path.join(self.result_dir, 'model.pt')
		self.plot_dir = os.path.join(self.result_dir, 'plots')
		self.loss_plot_path = os.path.join(self.plot_dir, 'loss.png')
		self.acc_plot_path = os.path.join(self.plot_dir, 'acc.png')

		self.results = {'train_loss': {}, 'train_acc': {}, 'train_acc_t5': {}, 'val_loss': {}, 'val_acc': {}, 'val_acc_t5': {}}
		self.best_epoch = 0

		#torch.set_default_device(P.DEVICE)
		self.epochs = self.config.get('epochs', 100)
		self.lr = self.config.get('lr', 1e-3)
		self.momentum = self.config.get('momentum', .9)
		self.wdecay = self.config.get('wdecay', 5e-4)
		self.sched_decay = self.config.get('sched_decay', .1)
		self.sched_milestones = self.config.get('sched_milestones', [])
		self.warmup_epochs = self.config.get('warmup_epochs', 0)
		self.warmup_gamma = self.config.get('warmup_gamma', 1)
		self.pretrain_path = self.config.get('pretrain_path', None)

		# Prepare quantizer
		self.quantizer = Quantizer(self.config)

		print("Loading dataset...")
		self.data_manager = utils.retrieve(self.config.get('data_manager', 'dataloaders.videodataset.UCF101DataManager'))(self.config, self.dataseed)
		self.train_dataloader = self.data_manager.load_trn()
		self.val_dataloader = self.data_manager.load_val()
		self.test_dataloader = self.data_manager.load_tst()
		print("Dataset {} loaded!".format(self.data_manager.dataset_name))

		print("Loading model...")
		utils.set_rng_seed(self.seed)
		self.model = utils.retrieve(self.config.get('model', 'models.R3D.R3D'))(self.config, num_classes=self.data_manager.num_classes)
		self.model.to(dtype=P.DTYPE)
		self.model.pre_params = nn.ParameterDict({n.replace('.', '/'): self.quantizer.param_to_pre_param(p) for n, p in self.model.named_parameters()}) if self.quantizer.pre_params_enabled() else None
		if self.pretrain_path is not None: # Initialize model from pre-trained dictionary if necessary
			model_path = self.pretrain_path.replace('<token>', self.token if self.token is not None else '')
			print("Initializing model from pre-trained dictionary at {}...".format(model_path))
			try: print(self.model.load_state_dict(utils.map_dtype(utils.load_dict(model_path), dtype=P.DTYPE), strict=False))
			except: print("WARNING: no model found in {}. Using model initialized from scratch.".format(model_path))
			if self.model.pre_params is not None:
				self.model.pre_params = nn.ParameterDict({n.replace('.', '/'): self.quantizer.param_to_pre_param(p) for n, p in self.model.named_parameters() if not n.startswith('pre_params')})
		if self.mode == 'test': # Load pretrained models for testing if necessary
			model_path = self.best_model_path
			print("Loading pre-trained model from {}...".format(model_path))
			try: self.model.load_state_dict(utils.map_dtype(utils.load_dict(model_path), dtype=P.DTYPE))
			except: print("WARNING: no model found in {}. Using untrained model for testing.".format(model_path))
		if self.model.pre_params is not None: self.model.pre_params.to(dtype=self.quantizer.preparam_precision)
		print("Model loaded!")

		# Resume training from previous checkpoint, or restart from scratch
		self.resume_epoch = 1
		self.epoch = self.resume_epoch
		self.criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
		self.test_evaluator = None
		self.optimizer = None
		self.scheduler = None
		if self.mode == 'train' and (restart or not os.path.exists(self.checkpoint_path)): # Setup optimization from scratch
			print("Preparing optimizer...")
			self.model.to(device=P.DEVICE)
			self.load_optimizer()
			print("Optimizer ready")
			print("Training from scratch...")
		else:
			if self.mode == 'train': # Load experiment state from checkpoint file
				print("Loading checkpoint from file: {}...".format(self.checkpoint_path))
				self.load_state_dict(utils.load_dict(self.checkpoint_path))
				print("Checkpoint loaded, resuming training from epoch {}...".format(self.resume_epoch))
			else: # Move model to device for testing
				self.model.to(device=P.DEVICE)

		print("Total model params: {:.2f}M".format(utils.count_params(self.model) / 1000000.0))

		self.tboard = None
		if self.mode == 'train': self.tboard = SummaryWriter('tboard/{}'.format(self.config_name), purge_step=self.resume_epoch)

		print("Experiment configuration ready!")

	def state_dict(self):
		return {
			'epoch': self.epoch + 1,
			'model_dict': self.model.state_dict(),
			'opt_dict': self.optimizer.state_dict(),
			'sched_dict': self.scheduler.state_dict(),
			'results': self.results,
			'best_epoch': self.best_epoch,
			'rng_state': utils.get_rng_state(),
			#'data_rng_state': self.data_manager.data_rng.get_state(),
		}

	def load_state_dict(self, state_dict):
		self.resume_epoch = state_dict['epoch']
		self.epoch = self.resume_epoch
		self.model.load_state_dict(state_dict['model_dict'])
		self.model.to(device=P.DEVICE)
		self.load_optimizer(state_dict)
		self.results = state_dict['results']
		self.best_epoch = state_dict['best_epoch']
		utils.set_rng_state(state_dict['rng_state'])
		#self.data_manager.data_rng.set_state(state_dict['data_rng_state'])

	def load_optimizer(self, saved_state=None):
		if hasattr(self.model, 'get_train_params'):
			train_params = self.model.get_train_params()
			if self.model.pre_params is not None:
				for tp in train_params: tp['params'] = [self.model.pre_params[n.replace('.', '/')] for n in tp['param_names']]
		else: train_params = list(self.model.pre_params.values()) if self.model.pre_params is not None else self.model.parameters()
		self.optimizer = optim.SGD(train_params, lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
		if saved_state is not None: self.optimizer.load_state_dict(saved_state['opt_dict'])
		#self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.sched_milestones, gamma=self.sched_decay)
		lr_lambda = WarmUpMultiStepLR(warmup_epochs=self.warmup_epochs, warmup_gamma=self.warmup_gamma, milestones=self.sched_milestones, gamma=self.sched_decay)
		self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=(saved_state['sched_dict']['last_epoch'] if saved_state is not None else -1))
		#if saved_state is not None: self.scheduler.load_state_dict(saved_state['sched_dict'])

	def save_results(self):
		# Flush tensorboard data
		self.tboard.flush()

		# Save training results to csv file
		utils.update_csv(self.results, self.result_path)

		# Also, save results as plot
		utils.save_trn_curve_plot(self.results['train_loss'], self.results['val_loss'], self.loss_plot_path, label='Loss')
		utils.save_trn_curve_plot(self.results['train_acc'], self.results['val_acc'], self.acc_plot_path, label='Accuracy')

		# If a new best model has been found at the current epoch, save it
		if self.epoch == self.best_epoch: utils.save_dict(copy.deepcopy(self.model.state_dict()), self.best_model_path)

		# Save experiment state to checkpoint file
		utils.save_dict(self.state_dict(), self.checkpoint_path)

	def extract_multi_crops(self, inputs):
		return inputs.reshape(-1, self.data_manager.input_channels, *inputs.shape[2:])

	def aggr_outputs(self, outputs, batch_count):
		outputs = outputs.reshape(batch_count, -1, outputs.shape[-1])
		outputs_smax = outputs.softmax(dim=-1)
		return outputs.mean(dim=1), outputs_smax.mean(dim=1)
		conf = torch.max(outputs_smax, 2)[0]
		#idx = (conf == torch.max(conf, 1, keepdim=True)[0])
		idx = torch.max(conf, 1)[1]
		return outputs[torch.arange(idx.shape[0]), idx], outputs_smax[torch.arange(idx.shape[0]), idx]

	def process_batch(self, batch):
		# move inputs and labels to the device the training is taking place on
		batch, idx = batch[:2], (batch[2] if len(batch) > 2 else None)
		inputs, labels = batch
		inputs, labels = inputs.to(device=P.DEVICE, dtype=P.DTYPE), labels.to(P.DEVICE)
		batch_count = inputs.shape[0]

		inputs = self.extract_multi_crops(inputs)
		outputs = self.model(inputs)
		outputs, outputs_smax = self.aggr_outputs(outputs, batch_count=batch_count)

		loss = self.criterion(outputs, labels)
		#outputs, outputs_smax = self.aggr_outputs(outputs, batch_count=batch_count)
		#outputs = outputs_smax
		hits = (labels == torch.max(outputs, 1)[1]).int().sum().item()
		hits_t5 = (labels.unsqueeze(-1) == outputs.topk(5, 1)[1]).int().sum().item()
		if self.test_evaluator is not None: self.test_evaluator.update(idx, outputs, labels)

		# Check for errors in the computation
		if math.isnan(loss):
			#utils.save_dict({'inputs': inputs, 'labels': labels, 'model': self.model.state_dict()}, '.debug/faulty.py')
			nan_params = [n for n, p in self.model.named_parameters() if torch.any(torch.isnan(p))]
			if len(nan_params) > 0:
				raise ValueError("Loss is nan. The following params in the model were found to be nan: " + str(nan_params))
			print("\nLoss is nan.")

		return loss, hits, hits_t5, batch_count

	def train_pass(self, dataloader):
		running_loss = 0.0
		running_hits = 0.0
		running_hits_t5 = 0.0
		running_count = 0

		self.model.train()

		for batch in tqdm(dataloader, ncols=80):
			batch_loss, batch_hits, batch_hits_t5, batch_count = self.process_batch(batch)

			if math.isnan(batch_loss): continue

			total_loss = batch_loss
			if hasattr(self.model, 'internal_loss'): total_loss = total_loss + self.model.internal_loss()
			total_loss = self.quantizer.rescale_loss(total_loss)

			self.model.zero_grad()
			self.optimizer.zero_grad()
			total_loss.backward()
			if self.model.pre_params is not None:
				for n, p in self.model.named_parameters():
					if not n.startswith('pre_params'): self.model.pre_params[n.replace('.', '/')].grad = self.quantizer.param_to_pre_param_grad(p)
			else:
				for p in self.model.parameters():
					p.grad = self.quantizer.param_to_pre_param_grad(p)
			self.optimizer.step()
			if self.model.pre_params is not None:
				with torch.no_grad():
					for n, p in self.model.named_parameters():
						if not n.startswith('pre_params'): p.data = self.quantizer.pre_param_to_param(self.model.pre_params[n.replace('.', '/')])

			running_loss += batch_loss.item() * batch_count
			running_hits += batch_hits
			running_hits_t5 += batch_hits_t5
			running_count += batch_count

		epoch_loss = running_loss / running_count
		epoch_acc = running_hits / running_count
		epoch_acc_t5 = running_hits_t5 / running_count

		return epoch_loss, epoch_acc, epoch_acc_t5

	def eval_pass(self, dataloader):
		running_loss = 0.0
		running_hits = 0.0
		running_hits_t5 = 0.0
		running_count = 0
		if self.test_evaluator is not None: self.test_evaluator.reset()

		self.model.eval()

		with torch.no_grad():
			for batch in tqdm(dataloader, ncols=80):
				batch_loss, batch_hits, batch_hits_t5, batch_count = self.process_batch(batch)

				if math.isnan(batch_loss): continue

				running_loss += batch_loss.item() * batch_count
				running_hits += batch_hits
				running_hits_t5 += batch_hits_t5
				running_count += batch_count

		epoch_loss = running_loss / running_count
		epoch_acc = running_hits / running_count
		epoch_acc_t5 = running_hits_t5 / running_count

		if self.test_evaluator is not None: epoch_loss, epoch_acc, epoch_acc_t5 = self.test_evaluator.compute()

		return epoch_loss, epoch_acc, epoch_acc_t5

	def run_eval(self):
		for dataset, dataloader in zip(['test', 'train', 'val'], [self.test_dataloader, self.train_dataloader, self.val_dataloader]):
			print("\nEVAL | Config {} | Iter {}".format(self.config_name, self.seed))
			print("Evaluating model on {} set...".format(dataset))
			self.test_evaluator = AggregateEvaluator(self.data_manager.tst_size, self.data_manager.num_classes, self.data_manager.clips_per_video*self.data_manager.crops_per_video,
													 self.criterion) if dataset == 'test' and self.data_manager.clips_per_video*self.data_manager.crops_per_video > 1 else None
			result_loss, result_acc, result_acc_t5 = self.eval_pass(dataloader)
			print("Results on {} set: loss {}, acc. {}, top-5 acc. {}".format(dataset, result_loss, result_acc, result_acc_t5))
			print("Saving results...")
			utils.update_iter_csv(self.seed, result_acc, os.path.join(self.test_result_path, dataset + '.csv'))
			utils.update_iter_csv(self.seed, result_acc_t5, os.path.join(self.t5_test_result_path, dataset + '.csv'))
		print("\nFinished!\n")

	def run_train(self):
		for epoch in range(self.resume_epoch, self.epochs + 1):
			start_time = timeit.default_timer()

			self.epoch = epoch
			print("\nEPOCH {}/{} | Config {} | Iter {}".format(self.epoch, self.epochs, self.config_name, self.seed))

			# Train phase
			print("Training...")
			train_loss, train_acc, train_acc_t5 = self.train_pass(self.train_dataloader)
			print("Train results: loss {}, acc. {}, top-5 acc. {}".format(train_loss, train_acc, train_acc_t5))
			self.results['train_loss'][self.epoch] = train_loss
			self.results['train_acc'][self.epoch] = train_acc
			self.results['train_acc_t5'][self.epoch] = train_acc_t5
			self.tboard.add_scalar("Loss/train", train_loss, epoch)
			self.tboard.add_scalar("Accuracy/train", train_acc, epoch)
			self.tboard.add_scalar("Accuracy_t5/train", train_acc_t5, epoch)

			# Validation phase
			print("Validating...")
			val_loss, val_acc, val_acc_t5 = self.eval_pass(self.val_dataloader)
			print("Validation results: loss {}, acc. {}, top-5 acc. {}".format(val_loss, val_acc, val_acc_t5))
			self.results['val_loss'][self.epoch] = val_loss
			self.results['val_acc'][self.epoch] = val_acc
			self.results['val_acc_t5'][self.epoch] = val_acc_t5
			self.tboard.add_scalar("Loss/val", val_loss, epoch)
			self.tboard.add_scalar("Accuracy/val", val_acc, epoch)
			self.tboard.add_scalar("Accuracy_t5/val", val_acc_t5, epoch)

			if val_acc > self.results['val_acc'].get(self.best_epoch, 0): self.best_epoch = self.epoch
			print("Best validation epoch so far {}".format(self.best_epoch))
			print("with val results: loss {}, acc. {}, top-5 acc. {}".format(
				self.results['val_loss'][self.best_epoch], self.results['val_acc'][self.best_epoch], self.results['val_acc_t5'][self.best_epoch]))

			# Update LR schedule at the end of each epoch
			self.scheduler.step()
			print("Updated LR: {}".format(self.scheduler.get_last_lr()))

			# Save results after each epoch
			print("Saving results...")
			self.save_results()
			print("Results saved!")

			# Evaluate epoch duration
			if P.DEVICE != 'cpu': torch.cuda.synchronize(P.DEVICE)
			epoch_duration = timeit.default_timer() - start_time
			print("Epoch duration: " + utils.format_time(epoch_duration))
			print("Expected remaining time: " + utils.format_time((self.epochs - self.epoch) * epoch_duration))

		print("\nFinished!\n")


def run_experiment(config, mode, device, restart, seeds, dataseeds, tokens, datafolder, fragsize):
	# Override default params
	P.DEVICE = device
	P.DATASET_FOLDER = datafolder
	if fragsize is not None: os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:{}'.format(fragsize)
	os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(P.ASSETS_FOLDER, 'hub')
	torch.hub.set_dir(os.path.join(P.ASSETS_FOLDER, 'hub'))

	for iter, seed in enumerate(seeds):
		dataseed = dataseeds[iter % len(dataseeds)]
		token = tokens[iter % len(tokens)] if tokens is not None else None
		if 'train' in mode: Experiment(config, 'train', restart, seed, dataseed, token).run_train()
		if 'test' in mode: Experiment(config, 'test', restart, seed, dataseed, token).run_eval()

if __name__ == "__main__":
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=P.DEFAULT_CONFIG, help="The experiment configuration you want to run.")
	parser.add_argument('--mode', default=P.DEFAULT_MODE, choices=['train', 'test', 'traintest'], help="Whether you want to run a train or a test experiment.")
	parser.add_argument('--device', default=P.DEVICE, choices=P.AVAILABLE_DEVICES, help="The device you want to use for the experiment.")
	parser.add_argument('--restart', action='store_true', default=P.DEFAULT_RESTART, help="Whether you want to restart the experiment from scratch, overwriting previous checkpoints in the save path.")
	parser.add_argument('--seeds', nargs='*', type=int, default=P.DEFAULT_SEEDS, help="RNG seeds to use for multiple iterations of the experiment.")
	parser.add_argument('--dataseeds', nargs='*', type=int, default=P.DEFAULT_DATASEEDS, help="RNG seeds to use for data preparation for multiple iterations of the experiment.")
	parser.add_argument('--tokens', nargs='*', default=P.DEFAULT_TOKENS, help="A list of strings to be replaced in special configuration options.")
	parser.add_argument('--datafolder', default=P.DEFAULT_DATAFOLDER, help="The location of the dataset folder")
	parser.add_argument('--fragsize', default=None, help="GPU memory allocation size [MB]. Set it to a desired value to avoid fragmentation.")
	args = parser.parse_args()

	run_experiment(args.config, args.mode, args.device, args.restart, args.seeds, args.dataseeds, args.tokens, args.datafolder, args.fragsize)


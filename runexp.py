import argparse
import os
from tqdm import tqdm
import timeit
import math
import copy

import torch
from torch import nn, optim

import params as P
import utils


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
		self.test_result_path = os.path.join(self.iter_dir, 'results.csv')
		self.best_model_path = os.path.join(self.result_dir, 'model.pt')
		self.plot_dir = os.path.join(self.result_dir, 'plots')
		self.loss_plot_path = os.path.join(self.plot_dir, 'loss.png')
		self.acc_plot_path = os.path.join(self.plot_dir, 'acc.png')

		self.results = {'train_loss': {}, 'train_acc': {}, 'val_loss': {}, 'val_acc': {}}
		self.best_epoch = 0

		#torch.set_default_device(P.DEVICE)
		self.epochs = self.config.get('epochs', 100)
		self.lr = self.config.get('lr', 1e-3)
		self.momentum = self.config.get('momentum', .9)
		self.wdecay = self.config.get('wdecay', 5e-4)
		self.sched_decay = self.config.get('sched_decay', .1)
		self.sched_milestones = self.config.get('sched_milestones', [])
		self.pretrain_path = self.config.get('pretrain_path', None)

		self.stretch_h, self.stretch_v, self.stretch_g = self.config.get('stretch', (1, 1, 1))
		self.precision = self.config.get('precision', 'float32')
		if self.precision != 'float32' and P.DEVICE == 'cpu':
			raise RuntimeError("Only float32 precision is supported when using cpu device")
		if self.precision == 'float16': P.DTYPE = torch.float16
		if self.precision == 'float64': P.DTYPE = torch.float64
		self.qat = self.config.get('qat', False)
		self.preparam_precision = torch.float32 if self.qat else P.DTYPE

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
		with torch.no_grad(): self.model.pre_params = nn.ParameterDict({n.replace('.', '/'): self.stretch_h * p for n, p in self.model.named_parameters()}) if self.stretch_h != 1 or self.qat else None
		if self.pretrain_path is not None or self.mode == 'test': # Load pretrained models if necessary
			model_path = self.best_model_path if self.mode == 'test' else self.pretrain_path.replace('<token>', self.token)
			self.model.load_state_dict(utils.map_dtype(utils.load_dict(model_path), dtype=P.DTYPE))
		if self.model.pre_params is not None: self.model.pre_params.to(dtype=self.preparam_precision)
		print("Model loaded!")

		# Resume training from previous checkpoint, or restart from scratch

		self.resume_epoch = 1
		self.epoch = self.resume_epoch
		self.criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
		#self.criterion.to(device=P.DEVICE, dtype=P.DTYPE)
		self.optimizer = None
		self.scheduler = None
		if self.mode == 'train' and restart: # Setup optimization from scratch
			print("Preparing optimizer...")
			self.model.to(device=P.DEVICE)
			self.load_optimizer()
			print("Optimizer ready")
			print("Training from scratch...")
		else:
			if self.mode == 'train': # Load experiment state from checkpoint file
				print("Searching for checkpoint file: {}...".format(self.checkpoint_path))
				self.load_state_dict(utils.load_dict(self.checkpoint_path))
				print("Checkpoint loaded, resuming training from epoch {}...".format(self.resume_epoch))
			else: # Move model to device for testing
				self.model.to(device=P.DEVICE)

		print("Total model params: {:.2f}M".format(utils.count_params(self.model) / 1000000.0))

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
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.sched_milestones, gamma=self.sched_decay)
		if saved_state is not None: self.scheduler.load_state_dict(saved_state['sched_dict'])

	def save_results(self):
		# Save training results to csv file
		utils.update_csv(self.results, self.result_path)

		# Also, save results as plot
		utils.save_trn_curve_plot(self.results['train_loss'], self.results['val_loss'], self.loss_plot_path, label='Loss')
		utils.save_trn_curve_plot(self.results['train_acc'], self.results['val_acc'], self.acc_plot_path, label='Accuracy')

		# If a new best model has been found at the current epoch, save it
		if self.epoch == self.best_epoch: utils.save_dict(copy.deepcopy(self.model.state_dict()), self.best_model_path)

		# Save experiment state to checkpoint file
		utils.save_dict(self.state_dict(), self.checkpoint_path)

	def process_batch(self, batch):
		# move inputs and labels to the device the training is taking place on
		inputs, labels = batch
		inputs, labels = inputs.to(device=P.DEVICE, dtype=P.DTYPE), labels.to(P.DEVICE)

		outputs = self.model(inputs)

		loss = self.criterion(outputs, labels)
		preds = torch.max(outputs, 1)[1]
		hits = torch.sum(preds == labels).item()

		# Check for errors in the computation
		if math.isnan(loss):
			nan_params = [n for n, p in self.model.named_parameters() if torch.any(torch.isnan(p))]
			raise ValueError("Loss is nan. The following params in the model were found to be nan: " + str(nan_params))

		return loss, hits, inputs.size(0)

	def train_pass(self, dataloader):
		running_loss = 0.0
		running_hits = 0.0
		running_count = 0

		self.model.train()

		for batch in tqdm(dataloader, ncols=80):
			batch_loss, batch_hits, batch_count = self.process_batch(batch)

			total_loss = batch_loss
			if hasattr(self.model, 'internal_loss'): total_loss = total_loss + self.model.internal_loss()
			total_loss = self.stretch_v * total_loss

			self.model.zero_grad()
			self.optimizer.zero_grad()
			total_loss.backward()
			if self.model.pre_params is not None:
				for n, p in self.model.named_parameters():
					if not n.startswith('pre_params'): self.model.pre_params[n.replace('.', '/')].grad = ((self.stretch_g / self.stretch_v) * p.grad.to(dtype=self.preparam_precision) if p.grad is not None else None)
			else:
				for p in self.model.parameters():
					p.grad = ((self.stretch_g / self.stretch_v) * p.grad if p.grad is not None else None)
			self.optimizer.step()
			if self.model.pre_params is not None:
				with torch.no_grad():
					for n, p in self.model.named_parameters():
						if not n.startswith('pre_params'): p[:] = self.model.pre_params[n.replace('.', '/')] / self.stretch_h

			running_loss += batch_loss.item() * batch_count
			running_hits += batch_hits
			running_count += batch_count

		epoch_loss = running_loss / running_count
		epoch_acc = running_hits / running_count

		return epoch_loss, epoch_acc

	def eval_pass(self, dataloader):
		running_loss = 0.0
		running_hits = 0.0
		running_count = 0

		self.model.eval()

		with torch.no_grad():
			for batch in tqdm(dataloader, ncols=80):
				batch_loss, batch_hits, batch_count = self.process_batch(batch)

				running_loss += batch_loss.item() * batch_count
				running_hits += batch_hits
				running_count += batch_count

		epoch_loss = running_loss / running_count
		epoch_acc = running_hits / running_count

		return epoch_loss, epoch_acc

	def run_eval(self):
		print("\nTesting...")
		test_loss, test_acc = self.eval_pass(self.test_dataloader)
		print("Test results: loss {}, acc. {}".format(test_loss, test_acc))
		print("Saving results...")
		utils.update_iter_csv(self.seed, test_acc, self.test_result_path)
		print("\nFinished!\n")

	def run_train(self):
		for epoch in range(self.resume_epoch, self.epochs + 1):
			start_time = timeit.default_timer()

			self.epoch = epoch
			print("\nEPOCH {}/{} | Config {} | Iter {}".format(self.epoch, self.epochs, self.config_name, self.seed))

			# Train phase
			print("Training...")
			train_loss, train_acc = self.train_pass(self.train_dataloader)
			print("Train results: loss {}, acc. {}".format(train_loss, train_acc))
			self.results['train_loss'][self.epoch] = train_loss
			self.results['train_acc'][self.epoch] = train_acc

			# Validation phase
			print("Validating...")
			val_loss, val_acc = self.eval_pass(self.val_dataloader)
			print("Validation results: loss {}, acc. {}".format(val_loss, val_acc))
			self.results['val_loss'][self.epoch] = val_loss
			self.results['val_acc'][self.epoch] = val_acc

			if val_acc > self.results['val_acc'].get(self.best_epoch, 0): self.best_epoch = self.epoch
			print("Best validation epoch so far {}".format(self.best_epoch))
			print("with val results: loss {}, acc. {}".format(
				self.results['val_loss'][self.best_epoch], self.results['val_acc'][self.best_epoch]))

			# Update LR schedule at the end of each epoch
			self.scheduler.step()

			# Save results after each epoch
			print("Saving results...")
			self.save_results()
			print("Results saved!")

			epoch_duration = timeit.default_timer() - start_time
			print("Epoch duration: " + utils.format_time(epoch_duration))
			print("Expected remaining time: " + utils.format_time((self.epochs - self.epoch) * epoch_duration))

		print("\nFinished!\n")


def run_experiment(config, mode, device, restart, seeds, dataseeds, tokens, datafolder, fragsize):
	# Override default params
	P.DEVICE = device
	P.DATASET_FOLDER = datafolder
	if fragsize is not None: os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:{}".format(fragsize)

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


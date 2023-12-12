import argparse
import os
from tqdm import tqdm
import timeit
import math
import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import params as P
import utils
from dataloaders.dataset import VideoDataset


class Experiment:
	def __init__(self, config_name, restart, seed, dataseed, token):
		self.config_name = config_name
		self.seed = seed
		self.dataseed = dataseed
		self.token = token
		print("\n****    Config {} | Iter {}    ****".format(self.config_name, self.seed))

		self.config = utils.retrieve(self.config_name)
		print("Data seed: {}".format(self.dataseed))
		print("Experiment configuration: {}".format(self.config))

		print("\nInitializing experiment...")

		self.result_dir = os.path.join(P.RESULT_FOLDER, self.config_name.replace('.', os.sep), 'iter{}'.format(self.seed))
		self.checkpoint_path = os.path.join(self.result_dir, 'checkpoint.pt.tar')
		self.result_path = os.path.join(self.result_dir, 'results.csv')
		self.best_model_path = os.path.join(self.result_dir, 'model.pt.tar')

		self.results = {'train_loss': {}, 'train_acc': {},
						'val_loss': {}, 'val_acc': {},
						'test_loss': {}, 'test_acc': {}}
		self.best_epoch = 0

		self.batch_size = self.config.get('batch_size', 32)
		self.eval_batch_size = self.config.get('eval_batch_size', self.batch_size)
		self.clip_len = self.config.get('clip_len', 16)
		self.resize_height = self.config.get('resize_height', 128)
		self.resize_width = self.config.get('resize_width', 171)
		self.crop_size = self.config.get('crop_size', 112)
		self.eval_clip_len = self.config.get('eval_clip_len', self.clip_len)
		self.epochs = self.config.get('epochs', 100)
		self.lr = self.config.get('lr', 1e-3)
		self.momentum = self.config.get('momentum', .9)
		self.wdecay = self.config.get('wdecay', 5e-4)
		self.sched_decay = self.config.get('sched_decay', .1)
		self.sched_milestones = self.config.get('sched_milestones', [])

		torch.set_default_device(P.DEVICE)
		self.stretch_h, self.stretch_v, self.stretch_g = self.config.get('stretch', (1, 1, 1))
		self.precision = self.config.get('precision', 'float32')
		if self.precision != 'float32' and P.DEVICE == 'cpu':
			raise RuntimeError("Only float32 precision is supported when using cpu device")
		if self.precision == 'float16': torch.set_default_tensor_type(torch.HalfTensor)
		if self.precision == 'float64': torch.set_default_tensor_type(torch.DoubleTensor)
		self.grad_thr_wrn = None

		print("Loading dataset...")
		utils.set_rng_seed(self.dataseed)
		self.data_rng = torch.Generator(device=P.DEVICE).manual_seed(self.dataseed)
		self.dataset = self.config.get('dataset', 'UCF-101')
		self.train_dataloader = DataLoader(VideoDataset(dataset=self.dataset, split='train', clip_len=self.clip_len, resize_height=self.resize_height, resize_width=self.resize_width, crop_size=self.crop_size, precision=self.precision), batch_size=self.batch_size, shuffle=True, num_workers=4, generator=self.data_rng)
		self.val_dataloader = DataLoader(VideoDataset(dataset=self.dataset, split='val', clip_len=self.eval_clip_len, resize_height=self.resize_height, resize_width=self.resize_width, crop_size=self.crop_size, precision=self.precision), batch_size=self.eval_batch_size, num_workers=4, generator=self.data_rng)
		self.test_dataloader = DataLoader(VideoDataset(dataset=self.dataset, split='test', clip_len=self.eval_clip_len, resize_height=self.resize_height, resize_width=self.resize_width, crop_size=self.crop_size, precision=self.precision), batch_size=self.eval_batch_size, num_workers=4, generator=self.data_rng)
		print("Dataset {} loaded!".format(self.dataset))

		print("Loading model...")
		utils.set_rng_seed(self.seed)
		self.model = utils.retrieve(self.config.get('model', 'models.R3D.R3D'))(self.config, num_classes=VideoDataset.num_classes(self.dataset))
		with torch.no_grad(): self.model.pre_params = nn.ParameterDict({n.replace('.', '/'): self.stretch_h * p for n, p in self.model.named_parameters()}) if self.stretch_h != 1 else None
		self.model.to(device=P.DEVICE, dtype=torch.get_default_dtype())
		print("Model loaded!")

		print("Preparing optimizer...")
		if hasattr(self.model, 'get_train_params'):
			train_params = self.model.get_train_params()
			if self.model.pre_params is not None:
				for tp in train_params: tp['params'] = [self.model.pre_params[n.replace('.', '/')] for n in tp['param_names']]
		else: train_params = list(self.model.pre_params.values()) if self.model.pre_params is not None else self.model.parameters()
		self.criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
		self.criterion.to(device=P.DEVICE, dtype=torch.get_default_dtype())
		self.optimizer = optim.SGD(train_params, lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.sched_milestones, gamma=self.sched_decay)
		print("Optimizer ready")

		self.resume_epoch = 1
		self.epoch = self.resume_epoch
		if restart:
			print("Training from scratch...")
		else:
			print("Searching for checkpoint file: {}...".format(self.checkpoint_path))
			self.load_state_dict(utils.load_dict(self.checkpoint_path, device=P.DEVICE))
			print("Checkpoint loaded, resuming training from epoch {}...".format(self.resume_epoch))

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
			'data_rng_state': self.data_rng.get_state(),
		}

	def load_state_dict(self, state_dict):
		self.resume_epoch = state_dict['epoch']
		self.epoch = self.resume_epoch
		self.model.load_state_dict(state_dict['model_dict'])
		self.optimizer.load_state_dict(state_dict['opt_dict'])
		self.scheduler.load_state_dict(state_dict['sched_dict'])
		self.results = state_dict['results']
		self.best_epoch = state_dict['best_epoch']
		utils.set_rng_state(state_dict['rng_state'])
		self.data_rng.set_state(state_dict['data_rng_state'])

	def save_results(self):
		# Save evaluation results to csv file
		utils.update_csv(self.results, self.result_path)

		# Save experiment state to checkpoint file
		utils.save_dict(self.state_dict(), self.checkpoint_path)

		# If a new best model has been found at the current epoch, save it
		if self.epoch == self.best_epoch:
			utils.save_dict(copy.deepcopy(self.model.state_dict()), self.best_model_path)

	def process_batch(self, batch):
		# move inputs and labels to the device the training is taking place on
		inputs, labels = batch
		inputs, labels = inputs.to(device=P.DEVICE, dtype=torch.get_default_dtype()), labels.to(P.DEVICE)

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
				if self.grad_thr_wrn is None:
					for n, p in self.model.named_parameters():
						if not n.startswith('pre_params'): self.model.pre_params[n.replace('.', '/')].grad = ((self.stretch_g / self.stretch_v) * p.grad if p.grad is not None else None)
				else:
					very_large_grads, very_small_grads = [], []
					for n, p in self.model.named_parameters():
						if not n.startswith('pre_params'):
							new_grad = ((self.stretch_g / self.stretch_v) * p.grad if p.grad is not None else None)
							if torch.any(new_grad > self.grad_thr_wrn * self.model.pre_params[n.replace('.', '/')]):
								very_large_grads.append(n)
							if torch.any(new_grad < self.model.pre_params[n.replace('.', '/')] / self.grad_thr_wrn):
								very_small_grads.append(n)
							self.model.pre_params[n.replace('.', '/')].grad = new_grad
					if len(very_large_grads) + len(very_small_grads) > 0: print("WRN", end='')
					if len(very_large_grads) > 0: print(" | Very large gradients detected: {}".format(very_large_grads), end='')
					if len(very_small_grads) > 0: print(" | Very small gradients detected: {}".format(very_small_grads), end='')
			else:
				for p in self.model.parameters():
					p.grad = ((self.stretch_g / self.stretch_v) * p.grad if p.grad is not None else None)
			self.optimizer.step()
			if self.model.pre_params is not None:
				with torch.no_grad():
					for n, p in self.model.named_parameters():
						if not n.startswith('pre_params'): p.data = self.model.pre_params[n.replace('.', '/')] / self.stretch_h

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

	def launch(self):
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

			# Test phase
			print("Evaluating model on the test set...")
			test_loss, test_acc = self.eval_pass(self.test_dataloader)
			print("Test results: loss {}, acc. {}".format(test_loss, test_acc))
			self.results['test_loss'][self.epoch] = test_loss
			self.results['test_acc'][self.epoch] = test_acc

			if val_acc > self.results['val_acc'].get(self.best_epoch, 0): self.best_epoch = self.epoch
			print("Best validation epoch so far {}".format(self.best_epoch))
			print("with val results: loss {}, acc. {}".format(
				self.results['val_loss'][self.best_epoch], self.results['val_acc'][self.best_epoch]))
			print("and test results: loss {}, acc. {}".format(
				self.results['test_loss'][self.best_epoch], self.results['test_acc'][self.best_epoch]))

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


if __name__ == "__main__":
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=P.DEFAULT_CONFIG, help="The experiment configuration you want to run.")
	parser.add_argument('--device', default=P.DEVICE, choices=P.AVAILABLE_DEVICES, help="The device you want to use for the experiment.")
	parser.add_argument('--restart', action='store_true', default=P.DEFAULT_RESTART, help="Whether you want to restart the experiment from scratch, overwriting previous checkpoints in the save path.")
	parser.add_argument('--seeds', nargs='*', type=int, default=[0], help="RNG seeds to use for multiple iterations of the experiment.")
	parser.add_argument('--dataseeds', nargs='*', type=int, default=[100], help="RNG seeds to use for data preparation for multiple iterations of the experiment.")
	parser.add_argument('--tokens', nargs='*', default=None, help="A list of strings to be replaced in special configuration options.")
	parser.add_argument('--fragsize', default=None, help="GPU memory allocation size [MB]. Set it to a desired value to avoid fragmentation.")
	args = parser.parse_args()

	# Override default params
	P.DEVICE = args.device
	if args.fragsize is not None: os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:{}".format(args.fragsize)

	for iter, seed in enumerate(args.seeds):
		dataseed = args.dataseeds[iter % len(args.dataseeds)]
		token = args.tokens[iter % len(args.tokens)] if args.tokens is not None else None
		Experiment(args.config, args.restart, seed, dataseed, token).launch()

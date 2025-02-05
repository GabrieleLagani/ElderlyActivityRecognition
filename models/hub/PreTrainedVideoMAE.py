import os

import torch
import torch.nn as nn
import transformers

import utils
import params as P


class PreTrainedVideoMAE(nn.Module):
	def __init__(self, config, num_classes):
		super().__init__()
		self.net = transformers.VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')

	def forward(self, x):
		return self.net(x.transpose(1, 2)).logits



if __name__ == '__main__':
	device = 'cuda:1'
	config = utils.retrieve('configs.pretrained.slowfast')
	print(config)
	model = PreTrainedVideoMAE(config, 400).to(device)
	print("Model loaded")
	inp = torch.randn([20, 3, 32, 224, 224], device=device)
	print("Input shape: {}".format(inp.shape))
	model.eval()
	with torch.no_grad():
		out = model(inp)
	print("Success: {}".format(out.shape))

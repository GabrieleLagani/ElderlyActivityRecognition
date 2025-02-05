import torch
import torch.nn as nn
from models.modutils import *

def test_linear():
	tolerance = 1e-4
	l1 = nn.Linear(16, 64)
	l2 = MultiHeadLinear(16, 64, heads=1, headsize=64, shared_map=False)
	l1.weight.data = l2.weight.reshape(64, 16)
	l1.bias.data = l2.bias.reshape(-1)

	x = torch.randn(20, 16)

	y1 = l1(x)
	y2 = l2(x)

	error = (y1 - y2).pow(2).mean()

	print(error)

	assert error < tolerance

def test_conv():
	tolerance = 1e-4
	l1 = nn.Conv3d(16, 64, kernel_size=3, stride=2, padding=1)
	l2 = MultiHeadConv3d(16, 64, heads=1, headsize=64, shared_map=False, kernel_size=3, stride=2, padding=1)
	l1.weight.data = l2.weight.reshape(64, 16, 3, 3, 3)
	l1.bias.data = l2.bias.reshape(-1)

	x = torch.randn(20, 16, 16, 32, 32)

	y1 = l1(x)
	y2 = l2(x)

	error = (y1 - y2).pow(2).mean()

	print(error)

	assert error < tolerance

def test_conv_grp():
	tolerance = 1e-4
	l1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, groups=4)
	l2 = MultiHeadConv3d(8, 16, heads=4, headsize=8, shared_map=False, kernel_size=3, stride=2, padding=1)
	print(l1.weight.shape, l2.weight.shape)
	l1.weight.data = l2.weight.reshape(64, 8, 3, 3, 3)
	l1.bias.data = l2.bias.reshape(-1)

	print(l1.weight.shape, l2.weight.shape)

	x = torch.randn(20, 32, 16, 32, 32)

	y1 = l1(x)
	#y2 = l2(x)
	y2 = l2(x.reshape(x.shape[0], 4, 8, *x.shape[2:]).transpose(1, 2).reshape(x.shape[0], -1, *x.shape[2:]))
	y2 = y2.reshape(y2.shape[0], 16, 4, *y2.shape[2:]).transpose(1, 2).reshape(y2.shape[0], -1, *y2.shape[2:])

	print(y1.shape, y2.shape)

	error = (y1 - y2).pow(2).mean()

	print(error)

	assert error < tolerance

def test_groups():
	tolerance = 1e-4
	x = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1, 1)
	w = torch.tensor([1., 1., 1., 1.]).reshape(2, 2, 1, 1, 1)

	y1 = torch.conv3d(x, w, groups=2)
	y2 = torch.tensor([3., 7.]).reshape(1, 2, 1, 1, 1)

	error = (y1 - y2).pow(2).mean()

	print(error)

	assert error < tolerance


if __name__ == '__main__':
	test_conv_grp()


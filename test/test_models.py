import torch

#from models.ViViT import ViViT
#from models.SwinTransformer3D import SwinTransformer3D
#from models.TubeViT import TubeViT


def test_c3d():
	from models.C3D import C3D
	config = {}
	net = C3D(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_r3d():
	from models.R3D import R3D
	config = {'layer_sizes': (2, 2, 2, 2)}
	net = R3D(config, 101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_r2plus1d():
	from models.R2Plus1D import R2Plus1D
	config = {'layer_sizes': (2, 2, 2, 2)}
	net = R2Plus1D(config, 101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_i3d():
	from models.I3D import I3D
	config = {}
	net = I3D(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_s3d():
	from models.S3D import S3D
	config = {}
	net = S3D(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_x3d():
	from models.X3D import X3D
	config = {'model_version': 'S'}
	net = X3D(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_slowfast():
	from models.SlowFast import SlowFast
	config = {}
	net = SlowFast(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_movinet():
	from models.MoViNet import MoViNet
	config = {'movinet_model': 'A0'}
	net = MoViNet(config, num_classes=400)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_stam():
	from models.STAM import STAM
	config = {'layer_sizes': (12, 12, 6, 8)}
	net = STAM(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_timesformer():
	from models.TimeSFormer import TimeSFormer
	config = {'layer_sizes': (12, 12)}
	net = TimeSFormer(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_vivit():
	from models.ViViT import ViViT
	config = {'vivit_model': 4, 'layer_sizes': (4, 12, 4, 12)}
	net = ViViT(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_tubevit():
	from models.TubeViT import TubeViT
	config = {'layer_sizes': (4, 12, 4, 12)}
	net = TubeViT(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_uniformer():
	from models.UniFormer import UniFormer
	config = {'layer_sizes': (2, 2, 2, 2)}
	net = UniFormer(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())

def test_swintransformer3d():
	from models.SwinTransformer3D import SwinTransformer3D
	config = {'layer_sizes': (2, 8, 2, 8, 2, 8, 2, 8)}
	net = SwinTransformer3D(config, num_classes=101)

	inputs = torch.rand(1, 3, 16, 112, 112)
	outputs = net(inputs)
	print(outputs.size())


import os

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t

import utils
import params as P


class PreTrainedSwinTransformer3D(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        torch.hub.set_dir(os.path.join(P.ASSETS_FOLDER, 'hub'))
        self.net = swin3d_t(weights='KINETICS400_V1')

    def forward(self, x):
        return self.net(x)



if __name__ == '__main__':
    device = 'cuda:1'
    config = utils.retrieve('configs.pretrained.slowfast')
    print(config)
    model = PreTrainedSwinTransformer3D(config, 400).to(device)
    print("Model loaded")
    inp = torch.randn([20, 3, 32, 224, 224], device=device)
    print("Input shape: {}".format(inp.shape))
    model.eval()
    with torch.no_grad():
        out = model(inp)
    print("Success: {}".format(out.shape))

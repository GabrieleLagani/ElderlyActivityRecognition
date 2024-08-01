import os

import torch
import torch.nn as nn

import utils
import params as P


class PreTrainedSlowFast(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        torch.hub.set_dir(os.path.join(P.ASSETS_FOLDER, 'hub'))
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

    def forward(self, x):
        return self.net([x[:, :, ::4], x])



if __name__ == '__main__':
    device = 'cuda:1'
    config = utils.retrieve('configs.pretrained.slowfast')
    print(config)
    model = PreTrainedSlowFast(config, 400).to(device)
    print("Model loaded")
    inp = torch.randn([20, 3, 32, 224, 224], device=device)
    print("Input shape: {}".format(inp.shape))
    model.eval()
    with torch.no_grad():
        out = model(inp)
    print("Success: {}".format(out.shape))

"""
	Requires Python fvcore package.
	Usage:
	  python perf.py --config <config> --device cpu --input_shape 16 112 112 --num_outputs 400
	Or also with Linux perf utility:
	  perf stat python perf.py --config <config> --device cpu --input_shape 16 112 112 --num_outputs 400
"""

import argparse
from fvcore.nn import FlopCountAnalysis
import torch
import utils
import params as P

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=P.DEFAULT_CONFIG, help="The experiment configuration you want to run.")
    parser.add_argument('--device', default=P.DEVICE, choices=P.AVAILABLE_DEVICES, help="The device you want to use for the experiment.")
    parser.add_argument('--input_shape', nargs=3, type=int, default=(16, 112, 112), help="The input shape for the experiment.")
    parser.add_argument('--num_inputs', type=int, default=20, help="The batch size for your experiment.")
    parser.add_argument('--num_outputs', type=int, default=101, help="The number of output channels of the model.")

    args = parser.parse_args()

    config = utils.retrieve(args.config)
    input_shape = [20, 3] + list(args.input_shape)
    model = utils.retrieve(config.get('model', 'models.CA3D.CA3D'))(config, args.num_outputs).to(args.device)
    print("Total model params: {:.2f}M".format(utils.count_params(model) / 1000000.0))
    flops = FlopCountAnalysis(model, torch.randn(input_shape, device=args.device))
    print("FLOPs: {:.2f}G".format(flops.total()/1000000000.0))
    print({k: '{:.2f}G'.format(v/1000000000.0) for k, v in flops.by_operator().items()})
    print("Input shape: {}".format(input_shape))
    print("Output shape: {}".format(list(model(torch.randn(input_shape, device=args.device)).shape)))
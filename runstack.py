import argparse

import params as P
import utils
from runexp import run_experiment

if __name__ == "__main__":
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--stack', default=P.DEFAULT_STACK, help="The experiment stack you want to run.")
	parser.add_argument('--mode', default=P.DEFAULT_MODE, help="Whether you want to run a train or a test experiment.")
	parser.add_argument('--device', default=P.DEVICE, choices=P.AVAILABLE_DEVICES, help="The device you want to use for the experiment.")
	parser.add_argument('--restart', action='store_true', default=P.DEFAULT_RESTART, help="Whether you want to restart the experiment from scratch, overwriting previous checkpoints in the save path.")
	parser.add_argument('--seeds', nargs='*', type=int, default=P.DEFAULT_SEEDS, help="RNG seeds to use for multiple iterations of the experiment.")
	parser.add_argument('--dataseeds', nargs='*', type=int, default=P.DEFAULT_DATASEEDS, help="RNG seeds to use for data preparation for multiple iterations of the experiment.")
	parser.add_argument('--tokens', nargs='*', default=P.DEFAULT_TOKENS, help="A list of strings to be replaced in special configuration options.")
	parser.add_argument('--datafolder', default=P.DEFAULT_DATAFOLDER, help="The location of the dataset folder")
	parser.add_argument('--fragsize', default=None, help="GPU memory allocation size [MB]. Set it to a desired value to avoid fragmentation.")
	args = parser.parse_args()
	
	stack = utils.retrieve(args.stack)

	failed_configs = []
	for exp in stack:
		print()
		try:
			run_experiment(config=exp.get('config', P.DEFAULT_CONFIG),
		               mode=exp.get('mode', args.mode),
		               device=exp.get('device', args.device),
		               restart=exp.get('restart', args.restart),
		               seeds=exp.get('seeds', args.seeds),
		               dataseeds=exp.get('dataseeds', args.dataseeds),
		               tokens=exp.get('tokens', args.tokens),
		               datafolder=exp.get('datafolder', args.datafolder),
		               fragsize=exp.get('fragsize', args.fragsize)
	               )
		except KeyboardInterrupt as e:
			print(e)
			exit()
		except Exception as e:
			print(e)
			failed_configs.append(exp.get('config', P.DEFAULT_CONFIG))
		if len(failed_configs) > 0: print("The following configurations did not complete successfully: {}".format(failed_configs))

	print("\nFinished!")


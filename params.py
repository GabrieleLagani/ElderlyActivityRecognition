import torch


DEFAULT_STACK = 'stacks.base.stack_base'
DEFAULT_CONFIG = 'configs.base.config_base'
DEFAULT_MODE = 'test'
DEFAULT_RESTART = False
DEFAULT_SEEDS = [0]
DEFAULT_DATASEEDS = [100]
DEFAULT_DATAFOLDER = 'datasets'
DEFAULT_TOKENS = None

AVAILABLE_DEVICES = ['cpu'] + ['cuda:' + str(i) for i in range(torch.cuda.device_count())]
DEVICE = 'cpu'
DTYPE = torch.float

DATASET_FOLDER = DEFAULT_DATAFOLDER
RESULT_FOLDER = 'results'
ASSETS_FOLDER = 'assets'

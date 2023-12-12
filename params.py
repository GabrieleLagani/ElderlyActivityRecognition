import torch


DEFAULT_CONFIG = 'configs.base.config_base'
DEFAULT_RESTART = False

AVAILABLE_DEVICES = ['cpu'] + ['cuda:' + str(i) for i in range(torch.cuda.device_count())]
DEVICE = 'cpu'

DATASET_FOLDER = 'datasets'
RESULT_FOLDER = 'results'

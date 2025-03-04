import os
import subprocess
import random
from tqdm import tqdm
import warnings

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import Kinetics
import numpy as np
import cv2

import params as P
import utils


# Resample frames in [T, C, H, W]-shaped tensor with a desired step. Possible add jittering, i.e. randomly perturbe the
# subsampling steps
def frame_subsample(x, step=1, jitter=None):
	if jitter is None: return x[::step]
	frame_count = x.shape[0]
	out_count = frame_count // step + (1 if frame_count % step > 0 else 0)
	idx = torch.randint(0, int(jitter*step), [out_count], device=x.device, dtype=torch.int) + step * torch.arange(0, out_count, device=x.device, dtype=torch.int)
	return x[idx[idx < frame_count]]

# Extract a temporal crop from a video clip at a given temporal offset.
def temporal_crop(x, idx, num_time_crops=None):
	if num_time_crops == 1 or num_time_crops is None: return x
	target_len = 2 * (x.shape[1] // (num_time_crops + 1))
	t_stride = (x.shape[1] - target_len) // (num_time_crops - 1)
	return x[:, idx*t_stride:idx*t_stride + target_len]

# Extract a spatial crop from a video clip at a given horizontal or vertical offset, depending if the largest dimension
# is horizontal or vertical, respectively.
def spatial_crop(x, idx, num_space_crops=None):
	if num_space_crops == 1 or num_space_crops is None: return x
	target_size = min(x.shape[2], x.shape[3])
	v_stride = (x.shape[2] - target_size) // (num_space_crops - 1)
	h_stride = (x.shape[3] - target_size) // (num_space_crops - 1)
	return x[:, :, idx*v_stride:idx*v_stride + target_size, idx*h_stride:idx*h_stride + target_size]

# computes necessary clip length to read from video in order to extract a desired number of clips for multicrop testing
def clip_len_for_multicrop(clip_len, num_crops):
	return (num_crops + 1) * (clip_len // 2)

# This function takes a number of parameters related to video frame resizing, resampling, clipping, and produces a string
# representation of the overall operation, which is useful for saving/retrieving files related to a particular type of clipping
def clip_str_repr(clip_size=112, clip_len=16, clip_location='start', clip_step=1, min_clip_step=None, max_clip_step=None, auto_len=None, min_auto_len=None, max_auto_len=None, frame_jitter=None):
	s = '_s-{}_l-{}-{}_{}{}'.format(clip_size, clip_location, clip_len,
			  'x-{}'.format(clip_step) if isinstance(clip_step, int) else (
			  'x-{}-{}-{}'.format(clip_step, min_clip_step, max_clip_step) if clip_step == 'rand' else (
			  'x-{}-{}-{}_f-{}'.format(clip_step, min_clip_step, max_clip_step, auto_len) if clip_step == 'auto' else (
			  'x-{}-{}-{}_f-{}-{}-{}'.format(clip_step, min_clip_step, max_clip_step, auto_len, min_auto_len, max_auto_len)))),
			  '_j-{}'.format(frame_jitter) if frame_jitter is not None else ''
		)
	#return s

	s = ''
	if clip_size is not None: s += '_s-{}'.format(clip_size)
	if clip_len is not None: s += '_l-{}-{}'.format(clip_location, clip_len)
	if clip_step != 1:
		if isinstance(clip_step, int):
			s += '_x-{}'.format(clip_step)
		else:
			if clip_len == 'rand':
				s += '_x-{}-{}-{}'.format(clip_step, min_clip_step, max_clip_step)
			elif clip_step == 'auto':
				if auto_len is not None or clip_len is not None:
					s += '_x-{}-{}-{}_f{}'.format(clip_step, min_clip_step if min_clip_step is not None else 1, max_clip_step if max_clip_step is not None else 'max', auto_len if auto_len is not None else clip_len)
				elif min_clip_step is not None and min_clip_step != 1:
					s += '_x-{}'.format(min_clip_step)
			else:
				s += '_x-{}-{}-{}_f{}-{}-{}'.format(clip_step, min_clip_step, max_clip_step, auto_len, min_auto_len, max_auto_len)
	if frame_jitter is not None: s += '_j-{}'.format(frame_jitter)
	return s

def backend_str_repr(backend):
	backends = ['default', 'opencv', 'pyav', 'ffmpeg-python', 'ffmpeg', 'img']
	if backend.lower() not in backends:
		raise ValueError("Video backend should be one of {}, found {}".format(backends, backend))
	backend_str = {'ffmpeg-python': '_be-ffmpeg', 'ffmpeg': '_be-ffmpeg', 'img': '_be-img'}
	return backend_str.get(backend.lower(), '')

class KineticsDownloader(Kinetics):
	valid_classes = ['400', '600', '700']

	def __init__(self, root, num_classes='400', split='train'):
		if num_classes not in self.valid_classes:
			raise ValueError("Number of classes for Kinetics should be one of {}".format(self.valid_classes))
		self.num_classes = num_classes
		self.root = os.path.join(root, split)
		self.split = split
		self.split_folder = os.path.join(self.root, self.split) # Split folders are for example kinetics400/train/train
		self.num_download_workers = 1 # More than one gives a bug

	def download_if_needed(self):
		if not os.path.exists(self.split_folder):
			self.download_and_process_videos()


class VideoDatasetFolder(Dataset):
	def __init__(self, data_folder='datasets/ucf101', target_data_folder=None, frame_resize=None, frame_resample=1,
				 min_frame_resample=None, max_frame_resample=None, auto_resample_num_frames=80, frame_jitter=None, backend='opencv',
				 frames_per_clip=16, clip_location='start', space_between_frames=1,
				 min_space_between_frames=None, max_space_between_frames=None,
				 auto_frames=None, min_auto_frames=None, max_auto_frames=None,
				 clips_per_video=1, crops_per_video=1, transform=None, stop_on_invalid_frames=False, default_shape=None):
		self.data_folder = data_folder
		self.frame_resize = frame_resize
		self.frame_resample = frame_resample
		self.min_frame_resample = min_frame_resample
		self.max_frame_resample = max_frame_resample
		self.auto_resample_num_frames = auto_resample_num_frames
		self.frame_jitter = frame_jitter
		self.backend = backend
		self.resaving = backend_str_repr(self.backend) != '' or self.frame_resize is not None or self.frame_resample > 1
		self.target_data_folder = self.data_folder if not self.resaving else target_data_folder
		if self.resaving and self.target_data_folder is None:
			raise ValueError("A target data folder must be provided when video resaving is required, found 'None'")
		self.resaving = self.resaving and not os.path.exists(self.target_data_folder)
		self.frames_per_clip = frames_per_clip
		self.clip_location = clip_location # Possible values: 'start', 'center', 'random', or integer
		if self.clip_location not in ['start', 'center', 'random'] and not isinstance(self.clip_location, int):
			raise ValueError("'clip_location' should be one of 'start', 'center', 'random', or integer, but found {}".format(self.clip_location))
		self.space_between_frames = space_between_frames
		if self.space_between_frames not in ['auto', 'rand', 'auto-rand'] and not isinstance(self.space_between_frames, int):
			raise ValueError("'space_between_frames' should be one of 'auto', 'rand', 'auto-rand', or integer, but found {}".format(self.space_between_frames))
		self.min_space_between_frames = min_space_between_frames
		self.max_space_between_frames = max_space_between_frames
		if self.space_between_frames == 'rand' and (self.min_space_between_frames is None or self.max_space_between_frames is None):
			raise ValueError("'min_space_between_frames' and 'max_space_between_frames' should be provided when using mode 'rand' for 'space_between_frames', but found {}, {}".format(self.min_space_between_frames, self.max_space_between_frames))
		self.auto_frames = auto_frames
		self.min_auto_frames = min_auto_frames
		self.max_auto_frames = max_auto_frames
		self.clips_per_video = clips_per_video
		self.crops_per_video = crops_per_video
		self.transform = transform
		self.stop_on_invalid_frames = stop_on_invalid_frames
		self.default_shape = default_shape
		DATA_DESCRIPTOR_FILE = os.path.join(self.target_data_folder.replace(P.DATASET_FOLDER, os.path.join(P.ASSETS_FOLDER, 'datafiles')), 'datadesc.pt')

		data_dict = None
		try: # Try to load dataset information from file
			data_dict = utils.load_dict(DATA_DESCRIPTOR_FILE)
		except: # Read files in data folder, check integrity, and collect data information in internal data structures
			file_names, frame_counts, fps_counts, labels, classes, class_to_idx = [], [], [], [], [], {}
			num_files = 0
			print("Checking integrity of files in data folder {}...".format(self.data_folder))
			for i, action_name in enumerate(tqdm(sorted(os.listdir(self.data_folder)), ncols=80)):
				classes.append(action_name)
				class_to_idx[action_name] = i
				for file_name in sorted(os.listdir(os.path.join(self.data_folder, action_name))):
					path = os.path.join(self.data_folder, action_name, file_name)
					file_ok, frame_count, fps = self.check_integrity(path)
					if file_ok:
						file_names.append(os.path.join(action_name, file_name))
						frame_counts.append(frame_count)
						fps_counts.append(fps)
						labels.append(i)
					num_files += 1
			print("Number of files: {}, Number of valid files: {}".format(num_files, len(file_names)))
			data_dict = {'file_names': file_names, 'frame_counts': frame_counts, 'fps_counts': fps_counts, 'labels': labels, 'classes': classes, 'class_to_idx': class_to_idx}
			utils.save_dict(data_dict, DATA_DESCRIPTOR_FILE)

		self.file_names = data_dict['file_names']
		self.frame_counts = data_dict['frame_counts']
		self.fps_counts = data_dict['fps_counts']
		self.labels = data_dict['labels']
		self.classes = data_dict['classes']
		self.class_to_idx = data_dict['class_to_idx']

	def check_integrity(self, path):
		print("\n\n\33[2F\33[80C\33[s", end="", flush=True) # Save current cursor position. This is needed to clean possible error messages printed by opencv.

		capture = cv2.VideoCapture(path)
		file_ok = capture.isOpened()
		frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if file_ok else None
		fps = capture.get(cv2.CAP_PROP_FPS) if file_ok else None
		frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if file_ok else None
		frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if file_ok else None

		# Save video at reduced size if required. This is useful because reading large-size videos from disk is very
		# time-consuming, becoming a performance bottleneck during training. So, using resized videos gives a great
		# performance improvement. If resized data folder already exists, does not overwrite content. In order to
		# restore a corrupted directory, it is necessary to delete both the 'datadesc.pt' file and the 'target_data_folder'.
		if file_ok and self.resaving:
			os.makedirs(os.path.dirname(path.replace(self.data_folder, self.target_data_folder)), exist_ok=True)
			resized_height = int(self.frame_resize * frame_height / min(frame_height, frame_width)) if self.frame_resize is not None else frame_height
			resized_width = int(self.frame_resize * frame_width / min(frame_height, frame_width)) if self.frame_resize is not None else frame_width
			step = self.frame_resample
			if step == 'auto':
				step = max(frame_count // self.auto_resample_num_frames, 1)
				if self.min_frame_resample is not None:
					step = max(step, self.min_frame_resample)
				if self.max_frame_resample is not None:
					step = min(step, self.max_frame_resample)

			if self.backend.lower() == 'img':
				i, in_count, out_count = 0, 0, 0
				while i < frame_count:
					retaining, frame = capture.read()
					if frame is not None:
						if in_count % step == 0:
							save_path = os.path.join(path.replace(self.data_folder, self.target_data_folder).rsplit('.', 1)[0], '{}.jpg'.format(out_count))
							os.makedirs(os.path.dirname(save_path), exist_ok=True)
							frame = np.array(cv2.resize(frame, (resized_width, resized_height)), dtype=np.dtype('uint8'))
							cv2.imwrite(filename=save_path, img=frame)
							out_count += 1
						in_count += 1
					i += 1
				frame_count = out_count
				fps = fps // step

			elif self.backend.lower() == 'ffmpeg-python':
				import ffmpeg
				i = ffmpeg.input(path.replace(self.data_folder, self.target_data_folder))
				ffmpeg.output(i, path.replace(self.data_folder, self.target_data_folder),
			              **{'vf': 'scale={}:{}'.format(2*(resized_width//2), 2*(resized_height//2)),
			                 'r': fps // step,
			                 #'c:v': 'libx265', #'crf': 24
			                 }
		              ).overwrite_output().run()
				frame_count = frame_count // step
				fps = fps // step

			elif self.backend.lower() == 'ffmpeg':
				subprocess.run('ffmpeg -i {} -vf "scale={}:{}" -r {} '
				               #'-c:v libx265 '#-crf 24 '
				               '{}'.format(path, 2*(resized_width//2), 2*(resized_height//2), fps // step, path.replace(self.data_folder, self.target_data_folder)))
				frame_count = frame_count // step
				fps = fps // step

			else:
				out = cv2.VideoWriter(path.replace(self.data_folder, self.target_data_folder), cv2.VideoWriter_fourcc(*'mp4v'), fps // step, (resized_width, resized_height), True)
				i, in_count, out_count = 0, 0, 0
				while i < frame_count:
					retaining, frame = capture.read()
					if frame is not None:
						if in_count % step == 0:
							frame = np.array(cv2.resize(frame, (resized_width, resized_height)), dtype=np.dtype('uint8'))
							out.write(frame)
							out_count += 1
						in_count += 1
					i += 1
				out.release()
				frame_count = out_count

		capture.release()
		if not file_ok: # Clean opencv error message
			print("\33[u\33[J", end='', flush=True)

		return file_ok and frame_count > 0, frame_count, fps

	def read_video_from_imgs(self, path, start=0, end=None, step=1):
		frames = []
		frame_count = len(os.listdir(path))
		if end is None: end = frame_count - 1
		for i in range(start, end + 1, step):
			idx = i
			if self.frame_jitter is not None: idx = i + random.randint(0, int(step*self.frame_jitter))
			if os.path.exists(os.path.join(path, '{}.jpg'.format(idx))):
				frame = np.array(cv2.imread(os.path.join(path, '{}.jpg'.format(idx))), dtype=np.dtype('uint8'))
				frames.append(frame)
		return frames

	def read_video(self, path, fps=None, start=0, end=None, step=1):
		"""
		Reads video at specified path, and returns it as a pytorch tensor with shape [T, C, H, W], where T is the number
		of time frames, C is the number of channels, and H and W are the height and width dimensions of each frame.
		:param path: Path to read the video from.
		:return: Tensor with shape [T, C, H, W].
		"""
		frames = []

		if self.backend.lower() == 'img':
			frames = self.read_video_from_imgs(path.rsplit('.', 1)[0], start, end, step)

		elif self.backend.lower() == 'pyav':
			from torchvision.io import read_video
			return frame_subsample(read_video(path, start_pts=start/fps, end_pts=end/fps, pts_unit='sec')[0].permute(0, 3, 1, 2), step, self.frame_jitter) # Requires PyAV, thread-unsafe, slower

		else:
			# Initialize a VideoCapture object to read video data into a numpy array
			capture = cv2.VideoCapture(path)

			frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
			i = start
			count = 0
			idx = 0
			capture.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
			if end is None: end = frame_count - 1
			retaining = True
			while (i < end + 1 and (retaining or not self.stop_on_invalid_frames)):
				retaining, frame = capture.read()
				if frame is not None:
					if count % step == 0:
						idx = count
						if self.frame_jitter is not None: idx = count + random.randint(0, int(step*self.frame_jitter))
					if count == idx: frames.append(np.array(frame, dtype=np.dtype('uint8')))
					count += 1
				i += 1

			# Release the VideoCapture once it is no longer needed
			capture.release()

		if len(frames) == 0:
			if self.default_shape is None: raise RuntimeError("No frames found in file {}".format(path))
			warnings.warn("No frames retrieved from file {}".format(path))
			frames = np.zeros(((end + 1 - start) // step, *self.default_shape), dtype=np.dtype('uint8'))
		frames = np.stack(frames, axis=0)

		return torch.tensor(frames, device='cpu', dtype=torch.uint8).permute(0, 3, 1, 2)

	def __getitem__(self, index):
		video_idx, clip_idx = divmod(index, self.clips_per_video*self.crops_per_video)
		clip_idx, crop_idx = divmod(clip_idx, self.crops_per_video)
		file_path = os.path.join(self.target_data_folder, self.file_names[video_idx])

		# Read video and label for the element at the specified index
		frame_count, fps = self.frame_counts[video_idx], self.fps_counts[video_idx]
		start, end, step, n_frames = self.clip_location, None, self.space_between_frames, self.frames_per_clip
		if n_frames is None: n_frames = frame_count
		if step == 'rand':
			step = random.randint(self.min_space_between_frames, self.max_space_between_frames)
		if step == 'auto':
			auto_frames = self.auto_frames if self.auto_frames is not None else n_frames
			step = max(frame_count // auto_frames, 1)
			if self.min_space_between_frames is not None:
				step = max(step, self.min_space_between_frames)
			if self.max_space_between_frames is not None:
				step = min(step, self.max_frame_resample)
		if step == 'auto-rand':
			auto_frames = self.auto_frames if self.auto_frames is not None else n_frames
			min_step = max(frame_count // self.max_auto_frames, 1) if self.max_auto_frames is not None else None
			if self.max_space_between_frames is not None:
				min_step = max(min_step, self.min_space_between_frames) if min_step is not None else self.min_space_between_frames
			if min_step is None: min_step = max(frame_count // auto_frames, 1)
			max_step = max(frame_count // self.min_auto_frames, 1) if self.min_auto_frames is not None else None
			if self.min_space_between_frames is not None:
				max_step = min(max_step, self.max_space_between_frames) if max_step is not None else self.max_space_between_frames
			if max_step is None: max_step = max(frame_count // n_frames, 1)
			step = random.randint(min_step, max_step) if min_step < max_step else random.randint(max_step, min_step)
		if self.clip_location == 'start': start = 0
		if self.clip_location == 'center': start = max(0, (frame_count - n_frames * step) // 2)
		if self.clip_location == 'random': start = random.randint(0, max(0, frame_count - n_frames * step))
		start = start + (clip_idx * frame_count)//self.clips_per_video
		start, end = (start, start + n_frames * step - 1) if start is not None and frame_count >= start + n_frames * step else (0, None)
		buffer = spatial_crop(self.read_video(file_path, fps, start, end, step), crop_idx, self.crops_per_video)
		label = torch.tensor(self.labels[video_idx], dtype=torch.int64, device='cpu')
		index = torch.tensor(index, dtype=torch.int64, device='cpu')

		# Apply transformations if necessary
		if self.transform is not None:
			buffer = self.transform(buffer)

		return buffer, label, index

	def __len__(self):
		return len(self.file_names) * self.clips_per_video * self.crops_per_video

class VideoDataManager:
	def __init__(self, config, dataseed):
		self.dataseed = dataseed
		self.num_workers = config.get('num_workers', 4)
		self.processing_device = P.DEVICE if config.get('workers_on_gpu', False) else 'cpu'
		self.processing_dtype = config.get('processing_dtype', 'uint8')
		if self.processing_dtype not in ['float32', 'uint8'] and self.processing_device == 'cpu':
			raise ValueError("Only dtypes float32 and uint8 are supported on device cpu, not {}".format(self.processing_dtype))
		self.batch_size = config.get('batch_size', 32)
		self.eval_batch_size = config.get('eval_batch_size', self.batch_size)
		self.preproc_batch_size = config.get('preproc_batch_size', self.eval_batch_size)
		self.augment_manager = config.get('augment_manager', None)
		self.augment_manager = utils.retrieve(self.augment_manager)(config) if self.augment_manager is not None else None
		self.input_size = config.get('input_size', 112)
		self.clip_len = config.get('clip_len', 16)
		self.clip_step = config.get('clip_step', 1)
		self.min_clip_step = config.get('min_clip_step', None)
		self.max_clip_step = config.get('max_clip_step', None)
		self.auto_len = config.get('auto_len', None)
		self.min_auto_len = config.get('min_auto_len', None)
		self.max_auto_len = config.get('max_auto_len', None)
		self.eval_clip_len = config.get('eval_clip_len', self.clip_len)
		self.eval_clip_step = config.get('eval_clip_step', self.clip_step)
		self.eval_min_clip_step = config.get('eval_min_clip_step', self.min_clip_step)
		self.eval_max_clip_step = config.get('eval_max_clip_step', self.max_clip_step)
		self.eval_auto_len = config.get('eval_auto_len', self.auto_len)
		self.eval_min_auto_len = config.get('eval_min_auto_len', self.min_auto_len)
		self.eval_max_auto_len = config.get('eval_max_auto_len', self.max_auto_len)
		self.preproc_clip_len = config.get('preproc_clip_len', self.eval_clip_len)
		self.preproc_clip_step = config.get('preproc_clip_step', self.eval_clip_step)
		self.preproc_min_clip_step = config.get('preproc_min_clip_step', self.eval_min_clip_step)
		self.preproc_max_clip_step = config.get('preproc_max_clip_step', self.eval_max_clip_step)
		self.preproc_auto_len = config.get('preproc_auto_len', self.eval_auto_len)
		self.preproc_min_auto_len = config.get('preproc_min_auto_len', self.eval_min_auto_len)
		self.preproc_max_auto_len = config.get('preproc_max_auto_len', self.eval_max_auto_len)
		self.frame_resize = config.get('frame_resize', None)
		self.frame_resample = config.get('frame_resample', 1)
		self.min_frame_resample = config.get('min_frame_resample', None)
		self.max_frame_resample = config.get('max_frame_resample', None)
		self.auto_resample_num_frames = config.get('auto_resample_num_frames', 80)
		self.frames_per_clip = config.get('frames_per_clip', self.clip_len)
		self.space_between_frames = config.get('space_between_frames', 1)
		self.min_space_between_frames = config.get('min_space_between_frames', None)
		self.max_space_between_frames = config.get('max_space_between_frames', None)
		self.frame_jitter = config.get('frame_jitter', None)
		self.auto_frames = config.get('auto_frames', None)
		self.min_auto_frames = config.get('min_auto_frames', None)
		self.max_auto_frames = config.get('max_auto_frames', None)
		self.eval_frames_per_clip = config.get('eval_frames_per_clip', self.eval_clip_len)
		self.eval_space_between_frames = config.get('eval_space_between_frames', self.space_between_frames)
		self.eval_min_space_between_frames = config.get('eval_min_space_between_frames', self.min_space_between_frames)
		self.eval_max_space_between_frames = config.get('eval_max_space_between_frames', self.max_space_between_frames)
		self.eval_frame_jitter = config.get('eval_frame_jitter', None)
		self.eval_auto_frames = config.get('eval_auto_frames', self.auto_frames)
		self.eval_min_auto_frames = config.get('eval_min_auto_frames', self.min_auto_frames)
		self.eval_max_auto_frames = config.get('eval_max_auto_frames', self.max_auto_frames)
		self.preproc_frames_per_clip = config.get('preproc_frames_per_clip', self.preproc_clip_len)
		self.preproc_space_between_frames = config.get('preproc_space_between_frames', self.eval_space_between_frames)
		self.preproc_min_space_between_frames = config.get('preproc_min_space_between_frames', self.eval_min_space_between_frames)
		self.preproc_max_space_between_frames = config.get('preproc_max_space_between_frames', self.eval_max_space_between_frames)
		self.preproc_frame_jitter = config.get('preproc_frame_jitter', None)
		self.preproc_auto_frames = config.get('preproc_auto_frames', self.eval_auto_frames)
		self.preproc_min_auto_frames = config.get('preproc_min_auto_frames', self.eval_min_auto_frames)
		self.preproc_max_auto_frames = config.get('preproc_max_auto_frames', self.eval_max_auto_frames)
		self.clip_location = config.get('clip_location', 'random')
		self.eval_clip_location = config.get('eval_clip_location', self.clip_location)
		self.preproc_clip_location = config.get('preproc_clip_location', self.eval_clip_location)
		self.stop_on_invalid_frames = config.get('stop_on_invalid_frames', False)
		self.backend = config.get('data_backend', 'default')
		self.multicrop_test = config.get('multicrop_test', False)
		self.clips_per_video = 1 if self.multicrop_test else config.get('clips_per_video', 1)
		self.crops_per_video = 1 if self.multicrop_test else config.get('crops_per_video', 1)
		self.cliplvl_split = config.get('cliplvl_split', False)
		self.timediff_enc = config.get('timediff_enc', False)

		# Setup RNG state for reproducibility
		prev_rng_state = utils.get_rng_state()
		utils.set_rng_seed(self.dataseed)

		# Set data augmentation on GPU, if required
		if self.processing_device.startswith('cuda'): mp.set_start_method('spawn')

		# Acquires information such as dataset size, number of classes, number of input channels
		self.acquire_dataset_metadata()

		# Define data transformations
		T = [] # Standard transformations for evaluation
		T.append(OffsetClip(self.eval_clip_len, offset=self.eval_clip_location, clip_step=self.eval_clip_step, min_clip_step=self.eval_min_clip_step, max_clip_step=self.eval_max_clip_step,
							auto_len=self.eval_auto_len, min_auto_len=self.eval_min_auto_len, max_auto_len=self.eval_max_auto_len)) # Take a fixed-size clip from a desired offset of the video
		if self.input_size != self.frame_resize: T.append(transforms.Resize(self.input_size, antialias=True)) # Resize shortest side of the frames to the desired size (unless input was already saved on disk at the desired size)
		T.append(transforms.CenterCrop(self.input_size)) # Take central square crop of desired size
		T.append(ToTensor()) # Map tensor values from integers in the range [0, 255] to floats in the range [0, 1], with shape [C, T, H, W]
		self.T = transforms.Compose(T)
		self.T_test = self.T # Standard transformations for testing
		if self.multicrop_test:
			T_test = [] # Standard transformations for evaluation
			T_test.append(OffsetClip(clip_len_for_multicrop(self.eval_clip_len, config.get('clips_per_video', 10)), offset=self.eval_clip_location, clip_step=self.eval_clip_step, min_clip_step=self.eval_min_clip_step, max_clip_step=self.eval_max_clip_step,
								auto_len=self.eval_auto_len, min_auto_len=self.eval_min_auto_len, max_auto_len=self.eval_max_auto_len)) # Take a fixed-size clip from a desired offset of the video
			if self.input_size != self.frame_resize: T_test.append(transforms.Resize(self.input_size, antialias=True)) # Resize shortest side of the frames to the desired size (unless input was already saved on disk at the desired size)
			T_test.append(ToTensor()) # Map tensor values from integers in the range [0, 255] to floats in the range [0, 1], with shape [C, T, H, W]
			self.T_test = transforms.Compose(T_test)
		T_preproc = [] # Standard transformations for preprocessing
		T_preproc.append(OffsetClip(self.preproc_clip_len, offset=self.preproc_clip_location, clip_step=self.preproc_clip_step, min_clip_step=self.preproc_min_clip_step, max_clip_step=self.preproc_max_clip_step,
							auto_len=self.preproc_auto_len, min_auto_len=self.preproc_min_auto_len, max_auto_len=self.preproc_max_auto_len)) # Take a fixed-size clip from a desired offset of the video
		if self.input_size != self.frame_resize: T_preproc.append(transforms.Resize(self.input_size, antialias=True)) # Resize shortest side of the frames to the desired size (unless input was already saved on disk at the desired size)
		T_preproc.append(transforms.CenterCrop(self.input_size)) # Take central square crop of desired size
		T_preproc.append(ToTensor()) # Map tensor values from integers in the range [0, 255] to floats in the range [0, 1], with shape [C, T, H, W]
		self.T_preproc = transforms.Compose(T_preproc)
		T_augm = [] # Randomized transformations with data augmentation for training
		if self.augment_manager is not None:
			T_augm.append(self.augment_manager.get_transform())
		else:
			T_augm.append(OffsetClip(self.clip_len, offset=self.clip_location, clip_step=self.clip_step, min_clip_step=self.min_clip_step, max_clip_step=self.max_clip_step,
									 auto_len=self.auto_len, min_auto_len=self.min_auto_len, max_auto_len=self.max_auto_len)) # Take a fixed-size clip from a random temporal location of the video
			if self.input_size != self.frame_resize: T_augm.append(transforms.Resize(self.input_size, antialias=True)) # Resize shortest side of the frames to the desired size (unless input was already saved on disk at the desired size)
			T_augm.append(transforms.RandomHorizontalFlip())
		T_augm.append(transforms.CenterCrop(self.input_size)) # Take central square crop of desired size
		T_augm.append(ToTensor()) # Map tensor values from integers in the range [0, 255] to floats in the range [0, 1], with shape [C, T, H, W]
		self.T_augm = transforms.Compose(T_augm)

		# Dataset splitting
		self.prepare_rnd_indices(config)

		# Add normalization transformation, with mean and std computed from the data
		self.mean, self.std = self.get_stats()
		T_norm = Normalize(self.mean, self.std)
		T_conv = Convert(dtype=self.processing_dtype, device=self.processing_device)
		self.T = transforms.Compose([T_conv, self.T, T_norm])
		self.T_test = transforms.Compose([T_conv, self.T_test, T_norm])
		self.T_augm = transforms.Compose([T_conv, self.T_augm, T_norm])

		# Add time-difference encoding transformation if necessary
		if self.timediff_enc:
			T_timediff = TimeDiff()
			self.T = transforms.Compose([self.T, T_timediff])
			self.T_test = transforms.Compose([self.T_test, T_timediff])
			self.T_augm = transforms.Compose([self.T_augm, T_timediff])

		# Take multiple crops (concatenated along the channel dimension) for testing if required
		if self.multicrop_test: self.T_test = transforms.Compose([self.T_test, MultiCrop(config.get('crops_per_video', 3), config.get('clips_per_video', 10))])

		# Load Datasets
		self.trn_set = self.load_split(self.get_trn_split(frames_per_clip=self.frames_per_clip, clip_location=self.clip_location, space_between_frames=self.space_between_frames,
						  min_space_between_frames=self.min_space_between_frames, max_space_between_frames=self.max_space_between_frames, frame_jitter=self.frame_jitter,
						  auto_frames=self.auto_frames, min_auto_frames=self.min_auto_frames, max_auto_frames=self.max_auto_frames,
						  transform=self.T_augm), self.batch_size, shuffle=True)
		self.val_set = self.load_split(self.get_val_split(frames_per_clip=self.eval_frames_per_clip, clip_location=self.eval_clip_location, space_between_frames=self.eval_space_between_frames,
						  min_space_between_frames=self.eval_min_space_between_frames, max_space_between_frames=self.eval_max_space_between_frames, frame_jitter=self.eval_frame_jitter,
						  auto_frames=self.eval_auto_frames, min_auto_frames=self.eval_min_auto_frames, max_auto_frames=self.eval_max_auto_frames,
						  transform=self.T), self.eval_batch_size)
		self.tst_set = self.load_split(self.get_tst_split(frames_per_clip=clip_len_for_multicrop(self.eval_frames_per_clip, config.get('clips_per_video', 10)) if self.multicrop_test else self.eval_frames_per_clip, clip_location=self.eval_clip_location, space_between_frames=self.eval_space_between_frames,
						  min_space_between_frames=self.eval_min_space_between_frames, max_space_between_frames=self.eval_max_space_between_frames, frame_jitter=self.eval_frame_jitter,
						  auto_frames=self.eval_auto_frames, min_auto_frames=self.eval_min_auto_frames, max_auto_frames=self.eval_max_auto_frames,
						  transform=self.T_test), self.eval_batch_size)

		# Restore previous RNG state
		utils.set_rng_state(prev_rng_state)

	def get_dataset_name(self):
		raise NotImplementedError

	def acquire_dataset_metadata(self):
		self.dataset_name = self.get_dataset_name()
		self.data_folder = os.path.join(P.DATASET_FOLDER, self.dataset_name)
		self.resaving = backend_str_repr(self.backend) != '' or self.frame_resize is not None or self.frame_resample > 1
		self.target_data_folder = self.data_folder if not self.resaving else self.data_folder + clip_str_repr(clip_size=self.frame_resize, clip_len=None, clip_location=None, clip_step=self.frame_resample, min_clip_step=self.min_frame_resample, max_clip_step=self.max_frame_resample, auto_len=self.auto_resample_num_frames) + backend_str_repr(self.backend)
		clip_opts = dict(clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video) if self.cliplvl_split else {}
		dataset = VideoDatasetFolder(self.data_folder, self.target_data_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
									 min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, backend=self.backend,
									 auto_resample_num_frames=self.auto_resample_num_frames, stop_on_invalid_frames=self.stop_on_invalid_frames, **clip_opts)
		self.dataset_size = len(dataset)
		self.input_channels = dataset[0][0].shape[1]
		self.num_classes = len(dataset.classes)

	def prepare_rnd_indices(self, config):
		self.splitsizes = config.get('splitsizes', (0.70, 0.15))
		if sum(self.splitsizes) > 1:
			raise ValueError("The sum of splitsizes must be <= 1, found {}".format(self.splitsizes))
		self.trn_size = int(self.splitsizes[0]*self.dataset_size)
		self.val_size = int(self.splitsizes[1]*self.dataset_size)
		self.tst_size = self.dataset_size - self.trn_size - self.val_size
		self.indices = list(range(self.dataset_size))
		random.shuffle(self.indices)
		self.trn_idx = self.indices[:self.trn_size]
		self.val_idx = self.indices[self.trn_size:-self.tst_size]
		self.tst_idx = self.indices[-self.tst_size:] if self.tst_size > 0 else self.val_idx # test on validation samples if tst_size == 0

	def get_trn_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		clip_opts = dict(clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video) if self.cliplvl_split else {}
		split = VideoDatasetFolder(self.data_folder, self.target_data_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, **clip_opts, transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		split = Subset(split, self.trn_idx)
		return split

	def get_val_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		clip_opts = dict(clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video) if self.cliplvl_split else {}
		split = VideoDatasetFolder(self.data_folder, self.target_data_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, **clip_opts, transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		split = Subset(split, self.val_idx)
		return split

	def get_tst_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		split = VideoDatasetFolder(self.data_folder, self.target_data_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video,
								   transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		split = Subset(split, self.tst_idx if self.cliplvl_split else [idx*self.crops_per_video*self.clips_per_video + i for idx in self.tst_idx for i in range(self.crops_per_video*self.clips_per_video)])
		return split

	def load_split(self, split, batch_size, shuffle=False):
		# self.data_rng = torch.Generator(device=self.processing_device).manual_seed(self.dataseed)
		return DataLoader(Subset(split, list(range(len(split)))), batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers) #, generator=self.data_rng) # Subset provides a thread-safe environment

	def load_trn(self):
		return self.trn_set

	def load_val(self):
		return self.val_set

	def load_tst(self):
		return self.tst_set

	def get_stats(self):
		STATS_FILE = os.path.join(self.target_data_folder.replace(P.DATASET_FOLDER, os.path.join(P.ASSETS_FOLDER, 'datafiles')), 'stats_seed-{}{}{}.pt'.format(self.dataseed,
			clip_str_repr(clip_size=self.input_size, clip_len=self.preproc_frames_per_clip, clip_location=self.preproc_clip_location, clip_step=self.preproc_space_between_frames, min_clip_step=self.preproc_min_space_between_frames, max_clip_step=self.preproc_max_space_between_frames, auto_len=self.preproc_auto_frames, min_auto_len=self.preproc_min_auto_frames, max_auto_len=self.preproc_max_auto_frames, frame_jitter=self.preproc_frame_jitter),
			clip_str_repr(clip_size=self.input_size, clip_len=self.preproc_clip_len, clip_location=self.preproc_clip_location, clip_step=self.preproc_clip_step, min_clip_step=self.preproc_min_clip_step, max_clip_step=self.preproc_max_clip_step, auto_len=self.preproc_auto_len, min_auto_len=self.preproc_min_auto_len, max_auto_len=self.preproc_max_auto_len)))
		stats_dict = None
		try:  # Try to load stats from file
			stats_dict = utils.load_dict(STATS_FILE)
		except:  # Stats file does not exist --> Compute statistics
			dataset = self.load_split(self.get_trn_split(frames_per_clip=self.preproc_frames_per_clip, clip_location=self.preproc_clip_location, space_between_frames=self.preproc_space_between_frames,
						  min_space_between_frames=self.preproc_min_space_between_frames, max_space_between_frames=self.preproc_max_space_between_frames,
						  auto_frames=self.preproc_auto_frames, min_auto_frames=self.preproc_min_auto_frames, max_auto_frames=self.preproc_max_auto_frames,
						  frame_jitter=self.preproc_frame_jitter, transform=self.T_preproc), self.preproc_batch_size)
			sum = torch.zeros(self.input_channels, device=P.DEVICE)
			sum_sq = torch.zeros(self.input_channels, device=P.DEVICE)
			count = 0
			print("Computing dataset statistics...")
			for batch in tqdm(dataset, ncols=80):
				inputs, _, _ = batch
				inputs = inputs.to(P.DEVICE)
				count += inputs.shape[0]
				sum += inputs.mean(dim=(2, 3, 4)).sum(0)
				sum_sq += (inputs**2).mean(dim=(2, 3, 4)).sum(0)
			mean = sum / count
			mean_sq = sum_sq / count
			std = (mean_sq - mean**2)**0.5
			stats_dict = {'mean': mean, 'std': std, 'rng_state': utils.get_rng_state()}
			utils.save_dict(stats_dict, STATS_FILE)
		utils.set_rng_state(stats_dict['rng_state']) # Restore RNG state as if stats were computed, even if they were loaded
		return stats_dict['mean'].tolist(), stats_dict['std'].tolist()

class HMDB51DataManager(VideoDataManager):
	def get_dataset_name(self):
		return 'hmdb51'

class UCF101DataManager(VideoDataManager):
	def get_dataset_name(self):
		return 'ucf101'

class KineticsDataManager(VideoDataManager):
	def __init__(self, config, dataseed):
		self.dataset_version = config.get('dataset_version', '400')
		super().__init__(config, dataseed)

	def get_dataset_name(self):
		return 'kinetics{}'.format(self.dataset_version)

	def acquire_dataset_metadata(self):
		self.dataset_name = self.get_dataset_name()
		self.data_folder = os.path.join(P.DATASET_FOLDER, self.dataset_name)
		self.resaving = self.frame_resize is not None or self.frame_resample > 1 or self.backend
		self.target_data_folder = self.data_folder if not self.resaving else self.data_folder + clip_str_repr(clip_size=self.frame_resize, clip_len=None, clip_location=None, clip_step=self.frame_resample, min_clip_step=self.min_frame_resample, max_clip_step=self.max_frame_resample, auto_len=self.auto_resample_num_frames) + backend_str_repr(self.backend)
		trn_split, val_split, tst_split = self.get_trn_split(), self.get_val_split(), self.get_tst_split()
		self.trn_size, self.val_size, self.tst_size = len(trn_split), len(val_split), len(tst_split) // (self.clips_per_video * self.crops_per_video)
		self.dataset_size = self.trn_size + self.val_size + self.tst_size
		self.input_channels = trn_split[0][0].shape[1]
		self.num_classes = int(self.dataset_version)

	def prepare_rnd_indices(self, config):
		pass

	def get_trn_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		KineticsDownloader(self.data_folder, num_classes=self.dataset_version, split='train').download_if_needed()
		trn_folder, target_trn_folder = os.path.join(self.data_folder, 'train', 'train'), os.path.join(self.target_data_folder, 'train')
		split = VideoDatasetFolder(trn_folder, target_trn_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		return split

	def get_val_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		KineticsDownloader(self.data_folder, num_classes=self.dataset_version, split='val').download_if_needed()
		val_folder, target_val_folder = os.path.join(self.data_folder, 'val', 'val'), os.path.join(self.target_data_folder, 'val')
		split = VideoDatasetFolder(val_folder, target_val_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		return split

	def get_tst_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		KineticsDownloader(self.data_folder, num_classes=self.dataset_version, split='test').download_if_needed()
		tst_folder, target_tst_folder = os.path.join(self.data_folder, 'test', 'test'), os.path.join(self.target_data_folder, 'test')
		split = VideoDatasetFolder(tst_folder, target_tst_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video,
								   transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		return split

class KineticsClipDataManager(KineticsDataManager):
	def get_dataset_name(self):
		return 'kinetics{}clp'.format(self.dataset_version)

	def acquire_dataset_metadata(self):
		self.dataset_name = self.get_dataset_name()
		self.data_folder = os.path.join(P.DATASET_FOLDER, self.dataset_name)
		self.resaving = backend_str_repr(self.backend) != '' or self.frame_resize is not None or self.frame_resample > 1
		self.target_data_folder = self.data_folder if not self.resaving else self.data_folder + clip_str_repr(clip_size=self.frame_resize, clip_len=None, clip_location=None, clip_step=self.frame_resample, min_clip_step=self.min_frame_resample, max_clip_step=self.max_frame_resample, auto_len=self.auto_resample_num_frames) + backend_str_repr(self.backend)
		clip_opts = dict(clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video) if self.cliplvl_split else {}
		dataset = VideoDatasetFolder(os.path.join(self.data_folder, 'train', 'train'), os.path.join(self.target_data_folder, 'train'), frame_resize=self.frame_resize, frame_resample=self.frame_resample,
									 min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, backend=self.backend,
									 auto_resample_num_frames=self.auto_resample_num_frames, stop_on_invalid_frames=self.stop_on_invalid_frames, **clip_opts)
		self.dataset_size = len(dataset)
		self.input_channels = dataset[0][0].shape[1]
		self.num_classes = len(dataset.classes)

	def prepare_rnd_indices(self, config):
		self.splitsizes = config.get('splitsizes', (0.70, 0.15))
		if sum(self.splitsizes) > 1:
			raise ValueError("The sum of splitsizes must be <= 1, found {}".format(self.splitsizes))
		self.trn_size = int(self.splitsizes[0]*self.dataset_size)
		self.val_size = int(self.splitsizes[1]*self.dataset_size)
		self.trn_size = self.trn_size + self.val_size
		self.val_size = 0
		self.tst_size = self.dataset_size - self.trn_size - self.val_size
		self.indices = list(range(self.dataset_size))
		random.shuffle(self.indices)
		self.trn_idx = self.indices[:self.trn_size]
		self.val_idx = self.indices[self.trn_size:-self.tst_size]
		self.tst_idx = self.indices[-self.tst_size:]

	def get_trn_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		KineticsDownloader(self.data_folder, num_classes=self.dataset_version, split='train').download_if_needed()
		trn_folder, target_trn_folder = os.path.join(self.data_folder, 'train', 'train'), os.path.join(self.target_data_folder, 'train')
		clip_opts = dict(clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video) if self.cliplvl_split else {}
		split = VideoDatasetFolder(trn_folder, target_trn_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, **clip_opts, transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		split = Subset(split, self.trn_idx)
		return split

	def get_tst_split(self, frames_per_clip=16, clip_location='start', space_between_frames=1,
					  min_space_between_frames=None, max_space_between_frames=None,
					  auto_frames=None, min_auto_frames=None, max_auto_frames=None,
					  frame_jitter=None, transform=None):
		KineticsDownloader(self.data_folder, num_classes=self.dataset_version, split='train').download_if_needed()
		tst_folder, target_tst_folder = os.path.join(self.data_folder, 'train', 'train'), os.path.join(self.target_data_folder, 'train')
		split = VideoDatasetFolder(tst_folder, target_tst_folder, frame_resize=self.frame_resize, frame_resample=self.frame_resample,
								   min_frame_resample=self.min_frame_resample, max_frame_resample=self.max_frame_resample, auto_resample_num_frames=self.auto_resample_num_frames,
								   backend=self.backend, frames_per_clip=frames_per_clip, clip_location=clip_location, space_between_frames=space_between_frames,
								   min_space_between_frames=min_space_between_frames, max_space_between_frames=max_space_between_frames,
								   auto_frames=auto_frames, min_auto_frames=min_auto_frames, max_auto_frames=max_auto_frames,
								   frame_jitter=frame_jitter, clips_per_video=self.clips_per_video, crops_per_video=self.crops_per_video,
								   transform=transform, stop_on_invalid_frames=self.stop_on_invalid_frames)
		split = Subset(split, self.tst_idx if self.cliplvl_split else [idx*self.crops_per_video*self.clips_per_video + i for idx in self.tst_idx for i in range(self.crops_per_video*self.clips_per_video)])
		return split


# Custom object to obtain data augmentation transformations
class AugmentManager:
	def __init__(self, config):
		self.T_augm = None

	def get_transform(self):
		return self.T_augm

# Custom object to obtain strong data augmentation transformations
class HardAugmentManager(AugmentManager):
	def __init__(self, config):
		super().__init__(config)

		clip_len = config.get('clip_len', 16)
		interframe_dist = config.get('interframe_dist', 1)
		max_interframe_dist = config.get('max_interframe_dist', None)
		input_size = config.get('input_size', 112)
		rel_delta = config.get('da_rel_delta', 1.25)
		time_scale_rel_delta = config.get('da_time_scale_rel_delta', 1.5)
		time_scale_p = config.get('da_time_scale_p', 0.3)
		timeflip_p = config.get('da_timeflip_p', 0.3)
		framejitter_step = config.get('da_framejitter_step', 4)
		framejitter_p = config.get('da_framejitter_p', 0.2)

		resize_p = config.get('da_resize_p', 0.3)
		rot_degrees = config.get('da_rot_degrees', 180) #30)
		rot_p = config.get('da_rot_p', 0.3)
		transl_p = config.get('da_transl_p', 0.5)

		jit_brightness = config.get('da_jit_brightness', 0.1)
		jit_contrast = config.get('da_jit_contrast', 0.1)
		jit_saturation = config.get('da_jit_saturation', 0.1)
		jit_hue = config.get('da_jit_hue', 20)
		jit_p = config.get('da_jit_p', 0.5)
		grayscale_p = config.get('da_grayscale_p', 0.2)

		T_augm = []
		if time_scale_rel_delta != 0 and time_scale_p > 0:
			T_augm.append(transforms.Resize(input_size*rel_delta), antialias=True)
			T_augm.append(transforms.RandomApply([
				RandomTimeScale(1/time_scale_rel_delta, time_scale_rel_delta)
			], p=time_scale_p))
		T_augm.append(transforms.RandomApply([
			JitteredFrameSampling(framejitter_step)
		], p=framejitter_p))
		T_augm.append(OffsetClip(clip_len, clip_step=interframe_dist, max_clip_step=max_interframe_dist, offset='random'))
		T_augm.append(transforms.RandomApply([
			transforms.ColorJitter(brightness=jit_brightness, contrast=jit_contrast, saturation=jit_saturation, hue=jit_hue/360)
		], p=jit_p))
		T_augm.append(transforms.RandomGrayscale(p=grayscale_p))
		T_augm.append(transforms.RandomHorizontalFlip())
		T_augm.append(RandomTimeFlip(p=timeflip_p))
		T_augm.append(transforms.RandomApply([
			transforms.Lambda(RandomResize(input_size/rel_delta, input_size*rel_delta))  # Random rescale
		], p=resize_p))
		T_augm.append(transforms.RandomApply([
			transforms.RandomRotation(degrees=rot_degrees, expand=True)
		], p=rot_p))
		T_augm.append(transforms.RandomApply([
			transforms.CenterCrop(input_size*rel_delta),  # Take a fixed-size central crop
			transforms.RandomCrop(input_size) # Take a smaller fixed-size crop at random position (random translation)
		], p=transl_p))
		self.T_augm = transforms.Compose(T_augm)

# Custom object to obtain lighter versions of data augmentation transformations
class LightAugmentManager(AugmentManager):
	def __init__(self, config):
		super().__init__(config)

		clip_len = config.get('clip_len', 16)
		interframe_dist = config.get('interframe_dist', 1)
		max_interframe_dist = config.get('max_interframe_dist', None)
		input_size = config.get('input_size', 112)
		rel_delta = config.get('da_rel_delta', 1.25)
		time_scale_rel_delta = config.get('da_time_scale_rel_delta', 1.25)
		time_scale_p = config.get('da_time_scale_p', 0.3)

		resize_p = config.get('da_resize_p', 0.3)
		rot_degrees = config.get('da_rot_degrees', 10)
		rot_p = config.get('da_rot_p', 0.3)
		transl_p = config.get('da_transl_p', 0.5)

		T_augm = []
		if time_scale_rel_delta != 0 and time_scale_p > 0:
			T_augm.append(transforms.Resize(input_size*rel_delta, antialias=True))
			T_augm.append(transforms.RandomApply([
				RandomTimeScale(1/time_scale_rel_delta, time_scale_rel_delta)
			], p=time_scale_p))
		T_augm.append(OffsetClip(clip_len, clip_step=interframe_dist, max_clip_step=max_interframe_dist, offset='random'))
		T_augm.append(transforms.RandomHorizontalFlip())
		T_augm.append(transforms.RandomApply([
			transforms.Lambda(RandomResize(input_size/rel_delta, input_size*rel_delta))  # Random rescale
		], p=resize_p))
		T_augm.append(transforms.RandomApply([
			transforms.RandomRotation(degrees=rot_degrees, expand=True)
		], p=rot_p))
		T_augm.append(transforms.RandomApply([
			transforms.CenterCrop(input_size*rel_delta),  # Take a fixed-size central crop
			transforms.RandomCrop(input_size) # Take a smaller fixed-size crop at random position (random translation)
		], p=transl_p))
		self.T_augm = transforms.Compose(T_augm)


# Custom transform for random input resize
class RandomResize:
	def __init__(self, min_size, max_size):
		self.min_size = int(min_size)
		self.max_size = int(max_size)

	def __call__(self, x):
		return TF.resize(x, random.randint(self.min_size, self.max_size), antialias=True)

# Custom transform for random video slowdown/fastforward
class RandomTimeScale:
	def __init__(self, min_speed_factor, max_speed_factor):
		self.min_speed_factor = min_speed_factor
		self.max_speed_factor = max_speed_factor

	def __call__(self, x):
		scale_factor = random.uniform(self.min_speed_factor, self.max_speed_factor)
		out = x.transpose(1, 0).unsqueeze(0)
		if not torch.is_floating_point(out): out = out.float()
		out = F.interpolate(out, scale_factor=(scale_factor, 1, 1), mode='trilinear')
		out = out.to(dtype=x.dtype).reshape(*out.shape[1:]).transpose(1, 0)
		return out

# Custom transform for clip extraction at a specified temporal offset from video
class OffsetClip:
	def __init__(self, clip_len, offset=0, clip_step=1, min_clip_step=None, max_clip_step=None, auto_len=None, min_auto_len=None, max_auto_len=None):
		self.clip_len = clip_len
		self.offset = offset
		if self.offset not in ['start', 'center', 'random'] and not isinstance(self.offset, int):
			raise ValueError("'offset' should be one of 'start', 'center', 'random', or integer, but found {}".format(self.offset))
		self.clip_step = clip_step
		if self.clip_step not in ['auto', 'rand', 'auto-rand'] and not isinstance(self.clip_step, int):
			raise ValueError("'clip_step' should be one of 'auto', 'rand', or 'auto-rand', or integer, but found {}".format(self.clip_step))
		self.min_clip_step = min_clip_step
		self.max_clip_step = max_clip_step
		if self.clip_step == 'rand' and (self.min_clip_step is None or self.max_clip_step is None):
			raise ValueError("'min_clip_step' and 'max_clip_step' should be specified when using mode 'rand' for 'clip_step', but found {}, {}".format(self.min_clip_step, self.max_clip_step))
		self.auto_len = auto_len
		self.min_auto_len = min_auto_len
		self.max_auto_len = max_auto_len

	def __call__(self, x):
		frame_count = x.shape[0]
		n_frames, offset, step = self.clip_len, self.offset, self.clip_step
		if step == 'rand':
			step = random.randint(self.min_clip_step, self.max_clip_step)
		if step == 'auto':
			auto_frames = self.auto_len if self.auto_len is not None else n_frames
			step = max(frame_count // auto_frames, 1)
			if self.min_clip_step is not None:
				step = max(step, self.min_clip_step)
			if self.max_clip_step is not None:
				step = min(step, self.max_clip_step)
		if step == 'auto-rand':
			auto_frames = self.auto_len if self.auto_len is not None else n_frames
			min_step = max(frame_count // self.max_auto_len, 1) if self.max_auto_len is not None else None
			if self.max_clip_step is not None:
				min_step = max(min_step, self.min_clip_step) if min_step is not None else self.min_clip_step
			if min_step is None: min_step = max(frame_count // auto_frames, 1)
			max_step = max(frame_count // self.min_auto_len, 1) if self.min_auto_len is not None else None
			if self.min_clip_step is not None:
				max_step = min(max_step, self.max_clip_step) if max_step is not None else self.max_clip_step
			if max_step is None: max_step = max(frame_count // n_frames, 1)
			step = random.randint(min_step, max_step) if min_step < max_step else random.randint(max_step, min_step)
		if offset == 'start': offset = 0
		if offset == 'center': offset = max(0, (x.shape[0] - self.clip_len * self.clip_step) // 2)
		if offset == 'random': offset = random.randint(0, max(0, x.shape[0] - self.clip_len * self.clip_step))
		offset = 0 if offset is None else offset
		return x[torch.arange(offset, offset + n_frames * step, step, device=x.device, dtype=torch.int) % frame_count]

# Custom transform for frame resampling with non-uniform random steps between frames
class JitteredFrameSampling:
	def __init__(self, step):
		self.step = step

	def __call__(self, x):
		return frame_subsample(x, step=self.step, jitter=self.step)

# Custom transform for flipping video tensor along the temporal dimension, so that the video looks played backward
class RandomTimeFlip:
	def __init__(self, p):
		self.p = p

	def __call__(self, x):
		if random.random() < self.p:
			return x.flip(0)
		return x

# Custom transform for mapping tensors to desired device and dtype
class Convert:
	def __init__(self, dtype=None, device=None):
		supported_dtypes = ['uint8', 'float16', 'float32', 'float64']
		if dtype not in [None] + supported_dtypes:
			raise ValueError("Unsupported dtype for conversion: {}. Supported dtypes are {}".format(dtype, supported_dtypes))
		if dtype not in [None, 'uint8', 'float32'] and device == 'cpu':
			raise ValueError("Only data types uint8 and float32 are supported on device cpu, not {}".format(dtype))
		self.dtype = None
		if dtype == 'uint8': self.dtype = torch.uint8
		if dtype == 'float16': self.dtype = torch.float16
		if dtype == 'float32': self.dtype = torch.float32
		if dtype == 'float64': self.dtype = torch.float64
		self.device = device

	def __call__(self, x: torch.Tensor):
		if torch.is_floating_point(x) and self.dtype == torch.uint8:
			x = x * 255
		elif not torch.is_floating_point(x) and self.dtype not in [torch.uint8, None]:
			x = x / 255
		return x.to(dtype=self.dtype, device=self.device)

# Custom transform for mapping uint tensors with values in the range [0, 255] and shape [T, C, H, W], to float tensors
# in the range [0, 1] and shape [C, T, H, W]
class ToTensor:
	def __init__(self):
		pass

	def __call__(self, x):
		if torch.is_floating_point(x): return x.transpose(1, 0)
		return x.transpose(1, 0).float() / 255

# Custom transform for normalizing video tensor with given mean and standard deviation values
class Normalize:
	def __init__(self, mean, std):
		if len(mean) != len(std):
			raise ValueError("Normalize expects mean and std vectors to have the same length, but found {} and {}".format(len(mean), len(std)))
		self.mean = mean
		self.std = std

	def __call__(self, x):
		for i in range(x.shape[0]):
			x[i] = (x[i] - self.mean[i]) / self.std[i]
		return x

# Custom transform for transforming a sequence of video frames by returning the difference between each frame and the previous frame
class TimeDiff:
	def __init__(self):
		pass

	def __call__(self, x):
		T = x.shape[0]
		for i in range(1, T):
			x[T - i] -= x[T - i - 1]
		return x

# Extract multiple crops from tensor and concatenate them along channel dimension
class MultiCrop:
	def __init__(self, num_space_crops=3, num_time_crops=10):
		self.num_space_crops = num_space_crops
		self.num_time_crops = num_time_crops

	def __call__(self, x):
		crops = torch.cat([temporal_crop(x, k, self.num_time_crops) for k in range(self.num_time_crops)], dim=0)
		crops = torch.cat([spatial_crop(crops, k, self.num_space_crops) for k in range(self.num_space_crops)], dim=0)
		return crops

# Custom transform for augmenting video frame channels with optical flow information
class OpticalFlow:
	"""
	Not yet implemented
	"""

	def __init__(self):
		raise NotImplemented

	def __call__(self, x):
		return x


import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

import params as P


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            precision (str): The float data type to use for loading the data: float16, float32, or float64 precision.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    @staticmethod
    def num_classes(dataset):
        if dataset == 'UCF-101': return 101
        if dataset == 'hmdb-51': return 51
        raise NotImplementedError("Dataset {} not available.".format(dataset))

    def __init__(self, dataset='UCF-101', split='train', clip_len=16, resize_height=128, resize_width=171, crop_size=112,
                 precision='float32', preprocess=False):
        self.dataset_dir = os.path.join(P.DATASET_FOLDER, dataset)
        self.preproc_dir = os.path.join(P.DATASET_FOLDER, 'preprocess', dataset)
        self.split = split
        self.precision = precision
        split_dir = os.path.join(self.preproc_dir, split)
        
        self.clip_len = clip_len
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(split_dir)):
            for fname in os.listdir(os.path.join(split_dir, label)):
                self.fnames.append(os.path.join(split_dir, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == 'UCF-101':
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb-51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

    def check_integrity(self):
        if not os.path.exists(self.dataset_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in preproc_dir
        if not os.path.exists(self.preproc_dir):
            return False
        elif not os.path.exists(os.path.join(self.preproc_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.preproc_dir, 'train'))):
            for video in os.listdir(os.path.join(self.preproc_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.preproc_dir, 'train', video_class, video),
                                          sorted(os.listdir(os.path.join(self.preproc_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        os.makedirs(os.path.join(self.preproc_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.preproc_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.preproc_dir, 'test'), exist_ok=True)

        # Split train/val/test sets
        for file in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.preproc_dir, 'train', file)
            val_dir = os.path.join(self.preproc_dir, 'val', file)
            test_dir = os.path.join(self.preproc_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.dataset_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(buffer[i], flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            #frame -= np.array([[[124.0, 116.0, 104]]])
            #frame /= np.array([[[58.0, 56.0, 57.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), dtype=np.dtype(self.precision))
        for i, frame_name in enumerate(frames):
            buffer[i] = np.array(cv2.imread(frame_name), dtype=np.dtype(self.precision))
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len) if buffer.shape[0] - clip_len > 0 else 0

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[[i%buffer.shape[0] for i in range(time_index, time_index + clip_len)],
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'train':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='UCF-101', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
import torch
import torch.utils.data as data
from PIL import Image

import os
import functools
import copy
import json

def pil_loader(path):
    # open path as file to avoid ResourceWarning

    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def video_loader(video_dir_path, frame_indices, image_loader):
    """
    Load video frames into an array
    """

    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_video_loader():
    return functools.partial(video_loader, image_loader = pil_loader)

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_class_labels(data):
    """
    Extracts the class labels from the dataset json file.
    """
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    """
    Extracts the video names and annotations from the dataset json file.
    Args:
        data: the json file content.
    Returns:
        video_names: a list of video names
        annotations: a dictionary of video name -> annotations for that video
    """
    video_names = []
    annotations = []

    for key, value in data['database'].items():

        this_subset = value['subset']

        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/' + key)

            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations

def make_dataset(video_path, sample_duration):
    """
    create a dataset from path of video and sample duration
    Args:
        Input: Path of video and duration of sample

        Output: Array of video dataset
    """

    dataset = []
    n_frames = len(os.listdir(video_path))

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    step = sample_duration

    for i in range(1, (n_frames - sample_duration + 1), step):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
        dataset.append(sample_i)

    return dataset


class Video(data.Dataset):
    """
    Applies transformations and returns the clip and target for the given index.
    """

    def __init__(self, video_path,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16,
                 get_loader=get_video_loader):
        self.data = make_dataset(video_path, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]['segment']
        return clip, target

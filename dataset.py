import os
import glob
import numpy as np
import random

import torch
from torch.utils.data import Dataset

class Normal_Loader(Dataset):
    """
    is_train:
    0 -> test
    1 -> train
    """

    def __init__(self, is_train = 1, root_dir = './'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.root_dir = root_dir

        if self.is_train == 1:
            data_dir = os.path.join(root_dir, 'train_normal.txt')
            with open(data_dir, 'r') as f:
                self.data = f.readlines()
        else:
            data_dir = os.path.join(root_dir, 'test_normal.txt')
            with open(data_dir, 'r') as f:
                self.data = f.readlines()

            random.shuffle(self.data)
            self.data = self.data[:-10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.root_dir+'all_rgbs', self.data[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.root_dir+'all_flows', self.data[idx][:-1]+'.npy'))
            concat_npy = np.concatenate((rgb_npy, flow_npy), axis=1)
            return concat_npy
        else:
            name, frames, gts = self.data[idx].split(' ')[0], int(self.data[idx].split(' ')[1]), int(self.data[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.root_dir+'all_rgbs', name+'.npy'))
            flow_npy = np.load(os.path.join(self.root_dir+'all_flows', name+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy, frames, gts


class Anomaly_Loader(Dataset):
    """
    is_train:
    0 -> test
    1 -> train
    """

    def __init__(self, is_train = 1, root_dir = './'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.root_dir = root_dir

        if self.is_train == 1:
            data_dir = os.path.join(root_dir, 'train_anomaly.txt')
            with open(data_dir, 'r') as f:
                self.data = f.readlines()
        else:
            data_dir = os.path.join(root_dir, 'test_anomaly.txt')
            with open(data_dir, 'r') as f:
                self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.root_dir+'all_rgbs', self.data[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.root_dir+'all_flows', self.data[idx][:-1]+'.npy'))
            concat_npy = np.concatenate((rgb_npy, flow_npy), axis=1)
            return concat_npy

        else:
            name, frames, gts = self.data[idx].split('|')[0], int(self.data[idx].split('|')[1]), self.data[idx].split('|')[3][:-1].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.root_dir+'all_rgbs', name+'.npy'))
            flow_npy = np.load(os.path.join(self.root_dir+'all_flows', name+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy, frames, gts

if __name__ == "__main__":
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))




import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from PIL import Image
import os
import cv2

# Instead of having the segmeneter to divide the dataset into segments
# I'll let the dataset module to handle it (makes it easier to load the dataset as well)


class Dataset(Dataset):
    """
    Custom Dataset for loading all images in Dataset Folder after segmentation.
    """

    def __init__(self, path) -> None:
        super().__init__()

        self.path = path

        self.listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(self.path):
            self.listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.listOfFiles)

    def __getitem__(self, index):

        image = cv2.imread(self.listOfFiles[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.show()


        return image



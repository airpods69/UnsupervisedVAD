import os

from dataset import *
from torch.utils.data import DataLoader


# Loading Normal train and test dataset
normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)
print("Normal Data Loaded")

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
print("Normal Data Loader Created")

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)
print("Anomaly Data Loaded")

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
print("Anomaly Data Loader Created")

print("Lenght of Normal Train Dataset: ", len(normal_train_dataset))

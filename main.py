from dataset import Dataset


path = './Dataset/Frames'

vids = Dataset(path)

print(vids.__getitem__(0))


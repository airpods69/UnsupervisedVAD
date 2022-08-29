import matplotlib.pyplot as plt
from video_dataset import VideoFrameDataset

path = './Dataset/Frames/'

dataset = VideoFrameDataset(path, './train.txt', num_segments=5, frames_per_segment=1, imagefile_template='frame_{:05d}.jpg', transform=None, test_mode=False
)

sample = dataset[0]  # take first sample of dataset 
frames = sample[0]   # list of PIL images
label = sample[1]    # integer label

for image in frames:
    plt.imshow(image)
    plt.title(label)
    plt.show()
    plt.pause(1)

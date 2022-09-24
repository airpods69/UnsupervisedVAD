import numpy as np
import matplotlib.pyplot as plt
from UnsupervisedVAD.video_dataset import VideoFrameDataset, ImglistToTensor
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import cv2
import numpy as np
import os
import json

path = './UnsupervisedVAD/Dataset/Frames/'

transfrom = transforms.Compose([
            ImglistToTensor(),
            transforms.CenterCrop((256,256)),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

dataset = VideoFrameDataset(path, './UnsupervisedVAD/train.txt', num_segments=100, frames_per_segment=16, imagefile_template='frame_{:05d}.jpg', transform=None, test_mode=False)
print("Dataset Formed")

segments = [[]] * len(dataset) # Get empty list of empty lists for each video in dataset
for j in range(len(dataset)):

    sample = dataset[j]
    frames = sample[0]
    # print(len(frames))

    segment = []

    for i in range(len(frames)//16):
        segment_i = frames[16 * i: 16 * (i + 1)]
        segment.append(segment_i)


    segments[j] = segment

segments = np.array(segments)
print("Segmenting done")


frameSize = (224, 224)
vid_tensors=[[]] * len(dataset)

#vid_tensors[j]= list of  tensors for the segments of video no. j

for i in range(len(segments)): #get_video
    lt=[]
    #print('video no.',i)
    vid=segments[i]
    frameset=np.array(vid) #array of segments
    for seg_index in range(0,len(frameset)):
        seg=frameset[seg_index]
        l=[]
        for k in range(0,len(seg)): #16 frames per segment
            #print('frame no.',k)
            pil_img=seg[k]
            cv_img=np.array(pil_img)
            cv_img=cv2.resize(cv_img,(112,112))
            l.append(cv_img)
       
        t=tuple(l)
        
        x = np.stack(t, axis = -1)
        y= np.transpose(x, (3,2,1,0)) #tensor for one segment
        lt.append(y)
    vid_tensors[i]=lt


vid_tensors=np.array(vid_tensors)
print("Converted into required format")

data_segments=[]

for j in range(0,len(vid_tensors)):
    for k in range(0,len(vid_tensors[j])):
        data_segments.append(vid_tensors[j][k])
print("Added all segments together")
print(len(data_segments))


video_path = '/content/outpy.avi'
output_file = '/content/outpy.npy'
count=0
ls=[]
for s in range(0, len(data_segments)):
    count+=1
    tens=data_segments[j]
    op_d={'video': tens, 'input': video_path, 'output': output_file}
    op_d_new=op_d
    op_d_new['video']=np.array(op_d['video']).tolist()
    ls.append(op_d_new)
    print('writing line ',count)

json_object=json.dumps(ls)
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
print("Converted to json")    

if not os.path.exists('./output_features/'):
  os.mkdir('./output_features/')
  
os.system('python ./video_feature_extractor/extract.py --jsn="sample.json" --type=3d --batch_size=1 --resnext101_model_path=/content/resnext101.pth') # basically running extract.py


import cv2
import os


def write_path_to_annotation(write_path, frameNo, file):

    class_name,video_name = write_path.split('/')[3], write_path.split('/')[4]
    file.write("{}/{} 1 {} 0\n".format(class_name, video_name, frameNo))


def frame_extractor(path, video, frames_path):
    """
    Extracts Frame from path to video
    Args:
        Input: Path to video

        Output: 16 Frames stored into Folder(Foldername: Dataset/Path/)
    """

    class_name = path.split('/')[-1]

    frames_dir = os.path.join(frames_path, f'Frames/{class_name}/{video}')

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    print(f"Path to Video is: {os.path.join(path, video)}")

    capture = cv2.VideoCapture(os.path.join(path, video))

    video_dir = os.path.join(path, video)

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Keeping track of Frames
    frameNo = 0
    frameCount = 0

    file = open('./train.txt', 'a')

    while(True):

        success, frame = capture.read() # Success -> if frame read successfully or not

        frameNo += 1
        frameCount += 1
        if success:
            
            write_path = os.path.join(frames_dir ,'frame_{:05d}.jpg'.format(frameNo))
            print(f"Wrote {write_path}")
            
            cv2.imwrite(write_path, frame)
        else:
            break


    print(f'Number of frames in {video} is {frameCount}')
    write_path_to_annotation(write_path = str(write_path), frameNo=frameCount, file=file)
    capture.release()


def video_selector(path):
    """
    Selects the video and sends it for frame extraction
    Input: Path
    Output: None
    """

    path_dataset = os.path.join(path, 'Datasets') # Change according to location


    for i in os.listdir(path_dataset):
        for j in os.listdir(os.path.join(path_dataset, i)):
            class_dir = os.path.join(path_dataset, i) # Class directory
            print(os.path.join(class_dir, j))
            frame_extractor(class_dir, j, path)



path = "./Dataset/"


# frame_extractor(path, video)
video_selector(path)
# print(os.system('pwd'))

print("Done")


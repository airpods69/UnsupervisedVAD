import cv2
import os

def frame_extractor(path, video, frames_path):
    """
    Extracts Frame from path to video
    Args:
        Input: Path to video

        Output: 16 Frames stored into Folder(Foldername: Dataset/Path/)
    """

    frames_dir = os.path.join(frames_path, f'Frames/{video}')

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

    while(True):

        success, frame = capture.read() # Success -> if frame read successfully or not

        frameNo += 1
        frameCount += 1
        if success:
            print(f"Wrote {os.path.join(frames_dir ,f'frame_{frameNo}.jpg')}")
            cv2.imwrite(os.path.join(frames_dir ,f'frame_{frameNo}.jpg'), frame)
        else:
            break


    print(f'Number of frames in {video} is {frameCount}')
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
            print(os.path.join(os.path.join(path_dataset, i), j))
            frame_extractor(os.path.join(path_dataset, i), j, path)



path = "./Dataset/"


# frame_extractor(path, video)
video_selector(path)
# print(os.system('pwd'))

print("Done")


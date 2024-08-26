import pandas as pd

import os
import pickle


with open('Annotations/train_split.pkl', 'rb') as file:
    videos_list = pickle.load(file)

root_dir = 'FINADiving_MTL_256s'

available_videos = []
for i, (directory, video_idx) in enumerate(videos_list):
    videos_directory = os.path.join(root_dir, directory, str(video_idx))
    if not os.path.exists(videos_directory):
        print(f'Error path:{os.path.join(videos_directory)} does not exist')
    else:
        available_videos.append((directory, video_idx))

with open('Annotations/train_split_filtered_missing.pkl', 'wb') as file:
    pickle.dump(available_videos, file)


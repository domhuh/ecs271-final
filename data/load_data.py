import pandas as pd
from urllib import request
import os
import numpy as np
from alive_progress import alive_bar
import imageio
import cv2

#Probably better to use argparse

indices_train = np.load("train_indices.npy")
# n_trains = 100
n_frames = 10
frame_size = 128

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

create_dir('train_raw')
create_dir('train_np')

# Extract/filter out training set
with open('train.txt','r') as f:
    train_links = f.read().splitlines()
data_ref = pd.read_csv ("tgif-v1.0.tsv", sep = '\t',  header=None)
train_labels = {}
with alive_bar(len(data_ref)) as bar:
    for i,[url,label] in data_ref.iterrows():
        if i in indices_train:
            request.urlretrieve(url, filename=f'train_raw/{i}')
            train_labels[i] = label
            gif = imageio.get_reader(f'train_raw/{i}')
            ret = []
            steps = np.linspace(0,len(gif)-1,n_frames)
            for j in steps: #keep/save as uint8 (preprocess after)
                frame = list(gif)[int(j)]
                if len(frame.shape)>2: #not already rgb
                    ret.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(frame_size,frame_size))) # convert to grayscle
                else:
                    ret.append(cv2.resize(frame,(frame_size,frame_size)))
              # out = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB) # convert to RGB
            np.save(f'train_np/{i}',ret)
        bar()
        
import json
with open("train_labels.json", 'w') as f:
    json.dump(train_labels,f)
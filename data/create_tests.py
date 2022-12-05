import pandas as pd
from urllib import request
import os
import numpy as np
from alive_progress import alive_bar
import imageio
import cv2

n_val = 20
n_test = 10
n_frames = 10
frame_size = 128

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


create_dir('val_raw')
create_dir('val_np')
create_dir('test_raw')
create_dir('test_np')

with open('corpus.txt','r') as f:
    corpus = f.read().splitlines()
    
#-----#
with open('val.txt','r') as f:
    val_links = f.read().splitlines()
data_ref = pd.read_csv ("tgif-v1.0.tsv", sep = '\t',  header=None)
val_ref = data_ref.loc[data_ref[0].isin(val_links)]

out = {}
for i,row in val_ref.iterrows():
    out[i] = len([i for i in row[1].split(' ') if i not in corpus])
best = [i for i,_ in sorted(out.items(), key = lambda x: x[1])[:n_val]]


val_labels= {}
with alive_bar(len(val_ref)) as bar:
    for i,row in val_ref.iterrows():
        if i in best:
            url, label = row
            request.urlretrieve(url, filename=f'val_raw/{i}')
            gif = imageio.get_reader(f'val_raw/{i}')
            ret = []
            steps = np.linspace(0,len(gif)-1,n_frames)
            for j in steps: #keep/save as uint8 (preprocess after)
                frame = list(gif)[int(j)]
                if len(frame.shape)>2: #not already rgb
                    ret.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(frame_size,frame_size))) # convert to grayscle
                else:
                    ret.append(cv2.resize(frame,(frame_size,frame_size)))
              # out = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB) # convert to RGB
            np.save(f'val_np/{i}',ret)
            val_labels[i] = label
        bar()

import json
with open("val_labels.json", 'w') as f:
    json.dump(val_labels,f)

#-----#

with open('test.txt','r') as f:
    val_links = f.read().splitlines()
data_ref = pd.read_csv ("tgif-v1.0.tsv", sep = '\t',  header=None)
val_ref = data_ref.loc[data_ref[0].isin(val_links)]

out = {}
for i,row in val_ref.iterrows():
    out[i] = len([i for i in row[1].split(' ') if i not in corpus])
best = [i for i,_ in sorted(out.items(), key = lambda x: x[1])[:n_test]]

val_labels= {}
with alive_bar(len(val_ref)) as bar:
    for i,row in val_ref.iterrows():
        if i in best:
            url, label = row
            request.urlretrieve(url, filename=f'test_raw/{i}')
            gif = imageio.get_reader(f'test_raw/{i}')
            ret = []
            steps = np.linspace(0,len(gif)-1,n_frames)
            for j in steps: #keep/save as uint8 (preprocess after)
                frame = list(gif)[int(j)]
                if len(frame.shape)>2: #not already rgb
                    ret.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(frame_size,frame_size))) # convert to grayscle
                else:
                    ret.append(cv2.resize(frame,(frame_size,frame_size)))
              # out = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB) # convert to RGB
            np.save(f'test_np/{i}',ret)
            val_labels[i] = label
        bar()

with open("test_labels.json", 'w') as f:
    json.dump(val_labels,f)
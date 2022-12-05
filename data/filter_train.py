import numpy as np
import pandas as pd
#Filtering
with open('train.txt','r') as f:
    train_links = f.read().splitlines()
data_ref = pd.read_csv ("tgif-v1.0.tsv", sep = '\t',  header=None)
train_ref = data_ref.loc[data_ref[0].isin(train_links)]

out = []
for i,row in train_ref.iterrows(): #just look at 200
    print(i,row[1])
    if int(input()) == 1:
        out.append(i)
    if i == 200:
        break
    
np.save("train_indices.npy",out)
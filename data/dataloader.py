# https://pytorch.org/text/main/transforms.html#berttokenizer
# Using https://arxiv.org/pdf/1609.08144.pdf to tokenize (wordpieces)

import torch, os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from tokenizers import BertWordPieceTokenizer
import torchvision.transforms as transforms
import json
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

class GIFDataset(Dataset):
    def __init__(self, dir, start=0): #dir = ['train', 'val', 'test']
        self.datalist = {os.path.basename(e).split(".")[0]:os.path.join(os.path.dirname(__file__),f'{dir}_np/{e}') 
                         for e in os.listdir(os.path.join(os.path.dirname(__file__),f"{dir}_np"))}
        tokenizer = BertWordPieceTokenizer(os.path.join(os.path.dirname(__file__),'./corpus_tokens.txt-vocab.txt'))
        with open(os.path.join(os.path.dirname(__file__),f"{dir}_labels.json"),'r') as f:
            self.labels = json.load(f)
        for k,v in self.labels.items():
            self.labels[k] = tokenizer.encode(v).ids
        self.maxlen = max([len(y) for y in self.labels.values()])
        self.all_possible_labels = []
        for i, annotation in self.labels.items():
            #y = np.tril([1 for i in range(len(y))])*y
            for j in range(start,len(annotation)):
                out = F.pad(torch.LongTensor(annotation[:j]),[0,self.maxlen-j], value = 4)
                next_token = annotation[j]
                self.all_possible_labels.append([i,out,next_token])

    def __len__(self):
        return len(self.all_possible_labels)
    def __getitem__(self, idx):
        k, annotation, next_token= self.all_possible_labels[idx]
        video = torch.cat([transform(img) for img in np.load(self.datalist[k])])
        return video, annotation, next_token, k

def get_dataloader(dir, batch_size):
    return DataLoader(GIFDataset(dir), batch_size=batch_size, shuffle=True)

# Another dataset for pretraining
class MaskedPretrainingGIFDataset(Dataset):
    def __init__(self, dir, start=0): #dir = ['train', 'val', 'test']
        self.datalist = {os.path.basename(e).split(".")[0]:os.path.join(os.path.dirname(__file__),f'{dir}_np/{e}') 
                         for e in os.listdir(os.path.join(os.path.dirname(__file__),f"{dir}_np"))}
        tokenizer = BertWordPieceTokenizer(os.path.join(os.path.dirname(__file__),'./corpus_tokens.txt-vocab.txt'))
        with open(os.path.join(os.path.dirname(__file__),f"{dir}_labels.json"),'r') as f:
            self.labels = json.load(f)
        for k,v in self.labels.items():
            self.labels[k] = tokenizer.encode(v).ids
        self.maxlen = max([len(y) for y in self.labels.values()])
        self.order = list(self.labels.keys())
        self.start = start
    def __len__(self):
        return len(self.order)
    def __getitem__(self, idx):
        k = self.order[idx]
        video = torch.cat([transform(img) for img in np.load(self.datalist[k])])
        annotation = self.labels[k]
        j = random.randint(self.start,len(annotation)-1)
        masked_annotation = F.pad(torch.LongTensor(annotation[:j]),[0,self.maxlen-j], value = 4)
        next_token = annotation[j]
        return video, masked_annotation, next_token, k

def get_masked_pretraining_dataloader(dir, batch_size):
    return DataLoader(MaskedPretrainingGIFDataset(dir), batch_size=batch_size, shuffle=True)


# Another dataset for pretraining
class PretrainingGIFDataset(Dataset):
    def __init__(self, dir, start=0): #dir = ['train', 'val', 'test']
        self.datalist = {os.path.basename(e).split(".")[0]:os.path.join(os.path.dirname(__file__),f'{dir}_np/{e}') 
                         for e in os.listdir(os.path.join(os.path.dirname(__file__),f"{dir}_np"))}
        tokenizer = BertWordPieceTokenizer(os.path.join(os.path.dirname(__file__),'./corpus_tokens.txt-vocab.txt'))
        with open(os.path.join(os.path.dirname(__file__),f"{dir}_labels.json"),'r') as f:
            self.labels = json.load(f)
        for k,v in self.labels.items():
            self.labels[k] = tokenizer.encode(v).ids
        self.maxlen = max([len(y) for y in self.labels.values()])
        self.order = list(self.labels.keys())
        self.start = start
    def __len__(self):
        return len(self.order)
    def __getitem__(self, idx):
        k = self.order[idx]
        video = torch.cat([transform(img) for img in np.load(self.datalist[k])])
        annotation = self.labels[k]
        masked_annotation = F.pad(torch.LongTensor(annotation),[0,self.maxlen-len(annotation)], value = 4)
        next_token = annotation[-1]
        return video, masked_annotation, next_token, k

def get_pretraining_dataloader(dir, batch_size):
    return DataLoader(PretrainingGIFDataset(dir), batch_size=batch_size, shuffle=True)
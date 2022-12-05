import os
import sys
path = "/"
for i in os.path.abspath("__file__").split("/"):
    path = os.path.join(path,i)
    if i=='ecs271-final':
        break
sys.path.append(path)
os.environ["PYTHONPATH"]=path

import torch
import torch.nn as nn
from models.main import *
from models.nlp_modules import *
from models.vision_modules import *
from data.dataloader import *
from loss import MILNCELoss, calculate_bleu
import os
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def pretrain(dir_path, n_epochs=100, masked=False,device='cuda'):
    model = MultiModalModule(n_frames=10, n_classes=vocab_size,device=device)

    if masked:
        train_dl = get_masked_pretraining_dataloader('train',batch_size=145)
    else:
        train_dl = get_pretraining_dataloader('train',batch_size=145)


    criterion = nn.CrossEntropyLoss()
    mil_nce = MILNCELoss(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay=0.1)

    pb = tqdm(range(n_epochs))

    train_losses = []

    for epoch in pb:
        closs = 0
        for _ in range(10):
            for video, masked_annotation, next_token, _ in train_dl:
                out = model(video.to(device),masked_annotation.to(device))
                loss = mil_nce(model.video_embed.to(device), model.text_embed.to(device))
                closs += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
          
        train_losses.append(closs)
    create_dir(dir_path)
    np.save(os.path.join(dir_path,'train_loss'),train_losses)
    torch.save(model.state_dict(),os.path.join(dir_path,'model.pth'))

def train(dir_path, n_epochs=1, pretrained_path=None, aux_loss = False, alpha=1e-3, device='cuda'):
    model = MultiModalModule(n_frames=10, n_classes=vocab_size,device=device)
    if pretrained_path!=None:
        model.load_state_dict(torch.load(pretrained_path))

    criterion = nn.CrossEntropyLoss()
    mil_nce = MILNCELoss(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=0.1)


    train_dl = get_dataloader('train', batch_size=128)
    val_dl = get_dataloader('val', batch_size=20)

    train_losses = []
    val_losses = []
    train_bleus = []
    val_bleus = []

    pb = tqdm(range(n_epochs))
    for epoch in pb:

        closs = 0
        for video, masked_annotation, next_token, _ in train_dl:
            out = model(video.to(device),masked_annotation.to(device))
            loss = criterion(out,next_token.to(device))
            closs += loss.item()
            if aux_loss:
                loss += alpha * mil_nce(model.video_embed.to(device), model.text_embed.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losses.append(closs)
        train_bleus.append(calculate_bleu(model,train_dl,train_dl.dataset.maxlen))
        
        with torch.no_grad():
            closs = 0 
            for video, masked_annotation, next_token, _ in val_dl:
                out = model(video.to(device),masked_annotation.to(device))
                loss = criterion(out,next_token.to(device))
                closs += loss.item()
            val_losses.append(closs)
            val_bleus.append(calculate_bleu(model,val_dl,train_dl.dataset.maxlen))

        pb.set_description(f"{epoch}: {round(train_losses[-1],3)} | {round(val_losses[-1],3)}")

    with torch.no_grad():
        test_dl = get_dataloader('test', batch_size=10)
        closs = 0 
        for video, masked_annotation, next_token, _ in test_dl:
            out = model(video.to(device),masked_annotation.to(device))
            loss = criterion(out,next_token.to(device))
            closs += loss.item()
        print('TESTING RESULTS:',closs, calculate_bleu(model,test_dl,train_dl.dataset.maxlen))

    create_dir(dir_path)
    np.save(os.path.join(dir_path,'train_loss'),train_losses)
    np.save(os.path.join(dir_path,'val_loss'),val_losses)
    np.save(os.path.join(dir_path,'train_bleu'),train_bleus)
    np.save(os.path.join(dir_path,'val_bleu'),val_bleus)
    torch.save(model.state_dict(),os.path.join(dir_path,'model.pth'))
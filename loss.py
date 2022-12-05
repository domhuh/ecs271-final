# https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/loss.py 
import torch
import torch.nn as nn
import os
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torchvision.transforms as transforms

class MILNCELoss(nn.Module):
    def __init__(self,device='cpu'):
        super(MILNCELoss, self).__init__()
        self.device = device
    def forward(self, video_embd, text_embd):
        x = torch.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * torch.eye(x.shape[0])[:,:,None].to(self.device)
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)

def calculate_bleu(model, dl, maxlen):
    device = model.device
    score = 0
    with torch.no_grad():
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
        seen = []
        for k in dl.dataset.all_possible_labels:
            if k[0] not in seen:
                seen.append(k[0])
                video = torch.cat([transform(img) for img in np.load(dl.dataset.datalist[k[0]])]).unsqueeze(0)
                prediction = torch.ones(size=(video.shape[0],maxlen)).long().to(device) * 4
                for i in range(maxlen):
                    tokens = model(video.to(device),prediction)
                    for j,pred in enumerate(prediction):
                        pred[i] = tokens.argmax(1)[j]
                reference = [dl.dataset.labels[k[0]]]
                candidate = prediction.cpu().numpy().tolist()[0]
                score += sentence_bleu(reference,candidate)
    return score
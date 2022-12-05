from .nlp_modules import nlp_model
from .vision_modules import S2D
import torch.nn as nn

class MultiModalModule(nn.Module):
	def __init__(self,n_frames, n_classes, device='cuda'):
		super().__init__()
		self.vision = S2D(n_frames)
		self.nlp = nlp_model()
		self.cross_encoder = nn.Sequential(nn.Linear(1024,1024),
										   nn.ReLU(),
										   nn.Linear(1024, n_classes),)
										   #nn.Softmax(1))
		self.device = device
		self.to(self.device)
	def forward(self,video,annotation):
		self.video_embed = self.vision(video)
		self.text_embed = self.nlp(annotation).pooler_output
		return self.cross_encoder(self.video_embed + self.text_embed)

from utils import *

n_epochs = 100
pretrain("pretrained_no_mask",n_epochs=n_epochs)
pretrain("pretrained_mask",n_epochs=n_epochs,masked=True)
train('baseline',n_epochs=n_epochs)
train('aux',aux_loss=True,n_epochs=n_epochs)
train('pretrain_no_mask',pretrained_path='pretrained_no_mask/model.pth',n_epochs=n_epochs)
train('pretrain_mask',pretrained_path='pretrained_mask/model.pth',n_epochs=n_epochs)
train('pretrain_aux_no_mask',aux_loss=True,pretrained_path='pretrained_no_mask/model.pth',n_epochs=n_epochs)
train('pretrain_aux_mask',aux_loss=True,pretrained_path='pretrained_mask/model.pth',n_epochs=n_epochs)
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))
plt.subplot(221)
plt.plot(np.load('baseline/train_loss.npy'),label="baseline")
plt.plot(np.load('aux/train_loss.npy'),label="aux")
plt.plot(np.load('pretrain_mask/train_loss.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_mask/train_loss.npy'),label="pretrained+aux (masked)")
plt.plot(np.load('pretrain_no_mask/train_loss.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_no_mask/train_loss.npy'),label="pretrained+aux (masked)")

plt.legend()
plt.xlabel('epochs')
plt.ylabel('cross entropy loss')
plt.title('training')

plt.subplot(222)
plt.plot(np.load('baseline/val_loss.npy'),label="baseline")
plt.plot(np.load('aux/val_loss.npy'),label="aux")
plt.plot(np.load('pretrain_mask/val_loss.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_mask/val_loss.npy'),label="pretrained+aux (masked)")
plt.plot(np.load('pretrain_no_mask/val_loss.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_no_mask/val_loss.npy'),label="pretrained+aux (masked)")

plt.legend()
plt.xlabel('epochs')
plt.ylabel('cross entropy loss')
plt.title('validation')

plt.subplot(223)
plt.plot(np.load('baseline/train_bleu.npy'),label="baseline")
plt.plot(np.load('aux/train_bleu.npy'),label="aux")
plt.plot(np.load('pretrain_mask/train_bleu.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_mask/train_bleu.npy'),label="pretrained+aux (masked)")
plt.plot(np.load('pretrain_no_mask/train_bleu.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_no_mask/train_bleu.npy'),label="pretrained+aux (masked)")

plt.yscale('log')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('cumulative bleu')
plt.title('training')

plt.subplot(224)
plt.plot(np.load('baseline/val_bleu.npy'),label="baseline")
plt.plot(np.load('aux/val_bleu.npy'),label="aux")
plt.plot(np.load('pretrain_mask/val_bleu.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_mask/val_bleu.npy'),label="pretrained+aux (masked)")
plt.plot(np.load('pretrain_no_mask/val_bleu.npy'),label="pretrained (masked)")
plt.plot(np.load('pretrain_aux_no_mask/val_bleu.npy'),label="pretrained+aux (masked)")

plt.yscale('log')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('cumulative bleu')
plt.title('validation')
plt.savefig("training")

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(15,15))

plt.plot(np.load('pretrained_mask/train_loss.npy'),label='masked')
plt.plot(np.load('pretrained_no_mask/train_loss.npy'),label='not masked')

plt.yscale('log')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('mil-nce loss')
plt.title('pretraining')
plt.savefig("pretraining")
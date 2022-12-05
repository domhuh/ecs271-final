# Following https://arxiv.org/pdf/2002.06353v3.pdf
# Basic tokenizer (maybe use if other method doesn't work but doesn't follow from paper)
from collections import defaultdict
import numpy as np
from tokenizers import BertWordPieceTokenizer

train_labels = np.load("train_labels.npy",allow_pickle=True)

#Create corpus from training set
corpus = []
for l in train_labels:
  corpus.extend(l.split(' '))
corpus = list(set(corpus))
with open("corpus.txt",'w') as f:
  f.write('\n'.join(corpus))


#Wordpiece tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)
tokenizer.train(files="corpus.txt", vocab_size=len(corpus), min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[EOS]'])
tokenizer.save_model('.','corpus_tokens.txt')


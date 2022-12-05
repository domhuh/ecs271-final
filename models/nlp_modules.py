from transformers import BertConfig, BertModel
import os
from functools import partial
with open(os.path.join(os.path.dirname(__file__),"../data/corpus_tokens.txt-vocab.txt"),'r') as f:
	vocab = f.read()

vocab_size = len(vocab.split('\n'))
config = BertConfig(vocab_size=vocab_size,
	    			hidden_size = 1024,
	    			intermediate_size = 1024,
					num_hidden_layers = 3,
					num_attention_heads = 4)

nlp_model = partial(BertModel,config)

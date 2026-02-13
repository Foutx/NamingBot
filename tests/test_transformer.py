import torch
import sys
sys.path.append('.')

from src.embedings import Embedings
from src.transformer import TransformerModel

from transformers import AutoTokenizer


embedings = Embedings()

vocab_size = 30522

model = TransformerModel(vocab_size)

text = 'I want to be a ML engineer'

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokens = tokenizer(text, return_tensors='pt')

input_tokens = tokens['input_ids']

x_emb = embedings.get_embs(input_tokens)

preds = model(x_emb)

print(f'Predictions: {preds}')
print(f'Shape of preds: {preds.shape}')

print(f'first 5 predictions of second word: {preds[0, 1, :5]}')
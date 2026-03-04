import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from src.embedings import Embedings


embs = Embedings()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

text = "I want to be a ML engineer"

tokens = tokenizer(text, return_tensors='pt')

input_ids = tokens['input_ids']

print(f"Tokens: {input_ids}")

words_vectors = embs.get_embs(input_ids)

print(f"All vectors: {words_vectors}")
from transformers import AutoModel

import torch
import torch.nn as nn


class Embedings(nn.Module):

    def __init__(self):
        
        super().__init__()

        bert = AutoModel.from_pretrained("distilbert-base-uncased")

        self.emb = bert.embeddings.word_embeddings
        self.emb_pos = bert.embeddings.position_embeddings

        self.emb.weight.requires_grad = False
        self.emb_pos.weight.requires_grad = False

        self.embed_dim = 768

    def get_embs(self, input_tokens):
        
        vectors = self.emb(input_tokens)
        positions = torch.arange(input_tokens.shape[1], device=input_tokens.device)
        pos_vectors = self.emb_pos(positions)

        return vectors + pos_vectors
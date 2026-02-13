import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, emb_dim = 768, nums_heads = 8, num_layers = 4):
        super().__init__()

        transform_layer = nn.TransformerEncoderLayer(
            d_model = emb_dim,
            nhead = nums_heads,
            dim_feedforward = emb_dim * 4,
            dropout = 0.1,
            activation = 'gelu',
            batch_first = True
        )

        self.transformer = nn.TransformerEncoder(transform_layer, num_layers)
        self.out_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, mask = None):
        
        len_sent = x.size(1)

        if mask is None:  
            mask = torch.triu(torch.ones(len_sent, len_sent) * float('-inf'), diagonal=1)

        mask = mask.to(x.device)

        x = self.transformer(x, mask=mask)

        preds = self.out_layer(x)

        return preds




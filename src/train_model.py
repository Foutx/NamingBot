import json

import torch
from torch.utils.data import DataLoader
from torch import optim

from transformer import TransformerModel
from embedings import Embedings
from data_loader import MovieDataset


BATCH_SIZE = 20
VOCAB_SIZE = 30522
NUM_EPOCH = 40

model = TransformerModel(VOCAB_SIZE)
embs = Embedings()

with open('../data/data.json', 'r') as f:
    texts = json.load(f) 

dataset = MovieDataset(texts)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.Adam((list(model.parameters()) + list(embs.parameters())), lr = 0.003)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
embs.to(device)

print(device)

for epoch in range(NUM_EPOCH):

    total_loss = 0

    for batch_idx, input_ids in enumerate(loader):

        input_ids = input_ids.to(device)

        embedings = embs.get_embs(input_ids)

        preds = model(embedings)

        shift_preds = preds[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = criterion(
            shift_preds.view(-1, VOCAB_SIZE), 
            shift_labels.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    print(f'Epoch {epoch}, Avg loss: {total_loss/len(loader):.4f}')
    torch.save(model.state_dict(), f'../model_weights/transformer_weights_{epoch}.pth')
    torch.save(embs.state_dict(), f'../model_weights/embeddings_weights_{epoch}.pth')
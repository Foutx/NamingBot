from torch.utils.data import Dataset

from transformers import AutoTokenizer


class MovieDataset(Dataset):
    def __init__(self, texts, max_length=512):

        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length
    
    def __len__(self):
        
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        tokens = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return tokens['input_ids'].squeeze(0)
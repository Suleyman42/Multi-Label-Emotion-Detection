import pandas as pd
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

def load_and_prepare_data(csv_path, tokenizer):
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    return EmotionDataset(texts, tokenizer)

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

# Optional packages
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    USE_ITERSTRAT = True
except ModuleNotFoundError:
    USE_ITERSTRAT = False

try:
    import nlpaug.augmenter.word as naw
    USE_NLPAUG = True
except ModuleNotFoundError:
    USE_NLPAUG = False

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "distilbert-base-uncased"
NUM_LABELS = 5 # anger, fear, joy, sadness, surprise
EPOCHS = 1
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2 # Simulate larger batch size

# Emotion labels
EMOTION_COLS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Model
class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_output))

# Main training function
def train_model(df_path: str, model_out_path: str = "emotion_classifier/model/trained_model.pt"):
    # Ensure parent folder exists
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    df = pd.read_csv(df_path)
    df = df[df[EMOTION_COLS].sum(axis=1) > 0].reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df[EMOTION_COLS].values

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = EmotionDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EmotionClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_out_path)
    print(f"✅ Model saved to {model_out_path}")

if __name__ == "__main__":
    train_model("data/processed/cleaned_track-a.csv")

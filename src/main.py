import pandas as pd
import numpy as np
import random
import html

import torch
from textblob import Word  # KEIN nltk
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import torch_optimizer as optim

from src.dataset import EmotionDataset
from src.model import EmotionModel
from src.trainer import FocalLoss, Trainer
from torch.utils.data import DataLoader

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Neue Version:
def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        word = Word(random_word)
        synonyms = word.synsets
        lemmas = set()
        for syn in synonyms:
            for lemma in syn.lemma_names():
                lemma_word = lemma.replace("_", " ")
                if lemma_word.lower() != random_word.lower():
                    lemmas.add(lemma_word.lower())
        if lemmas:
            synonym_word = random.choice(list(lemmas))
            new_words = [synonym_word if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def main():
    set_seed(42)

    df = pd.read_csv("../data/raw/track-a.csv")
    label_names = ['anger', 'fear', 'joy', 'sadness', 'surprise']

    df["text"] = df["text"].apply(html.unescape)

    # Optional augmentation
    df_aug = df.sample(frac=0.2, random_state=42).copy()
    df_aug["text"] = df_aug["text"].apply(lambda x: synonym_replacement(x, n=1))
    df = pd.concat([df, df_aug]).reset_index(drop=True)

    emotion_model = EmotionModel(model_name='roberta-large', num_labels=len(label_names))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df[label_names].values,
        test_size=0.20,
        random_state=42
    )

    train_dataset = EmotionDataset(train_texts, train_labels, emotion_model.tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, emotion_model.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = optim.AdamP(emotion_model.model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 15
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.2 * total_steps), num_training_steps=total_steps
    )

    loss_fn = FocalLoss(alpha=1, gamma=2)

    trainer = Trainer(
        model_wrapper=emotion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=15,
        patience=10,
        label_names=label_names
    )

    trainer.train()

if __name__ == "__main__":
    main()

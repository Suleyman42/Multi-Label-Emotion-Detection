import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import ast

# 1. Load data
try:
    df = pd.read_csv("../data/cleaned_data")
    df["label_vec"] = df["label_vec"].apply(ast.literal_eval)
    df["label_vec_tuple"] = df["label_vec"].apply(tuple)
except FileNotFoundError:
    raise FileNotFoundError("Data file not found. Please check the path.")
except Exception as e:
    raise Exception(f"Error loading data: {str(e)}")

stratify_column = df["label_vec_tuple"] if len(df["label_vec_tuple"].unique()) < len(df) else None

# 2. Train-test split - Removed all stratification-related code
X_train, X_test, y_train, y_test = train_test_split(
    df["Clean_Text"], 
    df["label_vec"],
    test_size=0.2, 
    random_state=42, 
)

# 3. Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
        
        self.encodings = tokenizer(
            list(texts), 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# 4. Initialize tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = EmotionDataset(X_train, y_train, tokenizer)
test_dataset = EmotionDataset(X_test, y_test, tokenizer)

# 5. Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# 7. Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 8. Train
trainer.train()

# 9. Save Model & Tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# 10. Evaluate
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)

# 11. Classification Report
from sklearn.metrics import classification_report
import numpy as np
outputs = trainer.predict(test_dataset)
y_pred = (torch.sigmoid(torch.tensor(outputs.predictions)) > 0.5).int().numpy()
y_true = np.array(y_test.tolist())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["anger", "fear", "joy", "sadness", "surprise"]))
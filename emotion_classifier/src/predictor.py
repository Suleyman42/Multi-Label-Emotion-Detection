import torch
from torch.utils.data import DataLoader

EMOTION_LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def make_predictions(model, dataset, batch_size=8):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loader = DataLoader(dataset, batch_size=batch_size)
    binary_predictions = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().tolist()
            binary_predictions.extend(preds)

    text_predictions = [
        [EMOTION_LABELS[i] for i, val in enumerate(pred) if val == 1]
        for pred in binary_predictions
    ]

    return {
        "binary": binary_predictions,
        "text": text_predictions
    }

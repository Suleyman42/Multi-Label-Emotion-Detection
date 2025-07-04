import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class EmotionModel:
    """
    Wrapper for RoBERTa for multi-label emotion classification.
    """

    def __init__(self, model_name='roberta-large', num_labels=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

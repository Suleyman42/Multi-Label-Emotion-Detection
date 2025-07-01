import torch
from transformers import AutoModel, AutoTokenizer

class EmotionClassifier(torch.nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=5):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled_output))

def load_model_and_tokenizer(model_path, model_name="distilbert-base-uncased", num_labels=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EmotionClassifier(model_name, num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, tokenizer

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import os

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for multi-label classification.
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        BCE_loss = self.bce(logits, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class Trainer:
    """
    Trainer class for training and evaluation.
    """

    def __init__(self, model_wrapper, train_loader, val_loader, loss_fn, optimizer, scheduler,
                 num_epochs=15, patience=10, label_names=None):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.patience = patience
        self.label_names = label_names if label_names else []

    def train(self):
        best_f1 = 0
        counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}"):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = self.loss_fn(logits, labels)
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = train_loss / len(self.train_loader)

            val_f1_mean, best_thresholds, avg_val_loss = self.evaluate()

            print(f"\nEpoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1_mean:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Best thresholds: {best_thresholds}")

            if val_f1_mean > best_f1:
                best_f1 = val_f1_mean
                counter = 0

                output_dir = "../outputs"
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, "best_roberta_large_focal.pt")

                self.model_wrapper.save(model_path)
                print(f" Best model saved at {model_path} (Val F1: {best_f1:.4f})")
            else:
                counter += 1
                if counter >= self.patience:
                    print(f" Early stopping at epoch {epoch + 1} (no improvement for {self.patience} epochs)")
                    break

        print("\n Training finished.")

    def evaluate(self):
        self.model.eval()
        val_labels_all = []
        val_preds_all = []
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = self.loss_fn(logits, labels)
                val_loss += loss.item()

                val_labels_all.append(labels.cpu().numpy())
                val_preds_all.append(logits.sigmoid().cpu().numpy())

        avg_val_loss = val_loss / len(self.val_loader)

        val_labels_all = np.vstack(val_labels_all)
        val_preds_all = np.vstack(val_preds_all)

        best_thresholds = []
        for i in range(len(self.label_names)):
            best_f1 = 0
            best_t = 0.5
            for t in np.arange(0.2, 0.8, 0.01):
                preds = (val_preds_all[:, i] >= t).astype(int)
                f1 = f1_score(val_labels_all[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds.append(best_t)

        val_preds_bin = np.zeros_like(val_preds_all)
        for i, t in enumerate(best_thresholds):
            val_preds_bin[:, i] = (val_preds_all[:, i] >= t).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels_all, val_preds_bin, average=None, zero_division=0
        )
        val_f1_mean = np.mean(f1)

        print(f"\nValidation Loss: {avg_val_loss:.4f}")
        for i, label in enumerate(self.label_names):
            print(f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

        return val_f1_mean, best_thresholds, avg_val_loss

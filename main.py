import os
import gc
import json
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm.auto import tqdm

try:
    import sentencepiece
except ImportError:
    !pip install sentencepiece transformers scikit-learn pandas tqdm matplotlib seaborn --quiet

BASE_DIR = "/content/drive/MyDrive/HW3/"
DATASET_PATH = os.path.join(BASE_DIR, "dataset/dataset.csv")
OUTPUT_FOLDER = "saved_deberta_base_2025_BEST"
OUT_DIR = os.path.join(BASE_DIR, OUTPUT_FOLDER)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 2025
MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 32
EPOCHS = 6
LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomBlock(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class SentimentConfig(PretrainedConfig):
    model_type = "hf-sentiment-classifier"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 3,
        head: str = "mlp",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        self.dropout = float(dropout)

class SentimentClassifier(PreTrainedModel):
    config_class = SentimentConfig

    def __init__(self, config: Optional[SentimentConfig] = None):
        if config is None:
            config = SentimentConfig()
        super().__init__(config)
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = getattr(self.encoder.config, "hidden_size", 768)

        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.head = CustomBlock(
            self.hidden_size, config.num_labels, dropout=config.dropout
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        enc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(
            self.encoder.config, "type_vocab_size", 0
        ) > 0:
            enc_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**enc_kwargs)

        if hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state[:, 0, :]
        else:
            feat = outputs[0][:, 0, :]

        feat = self.norm(feat)
        feat = self.dropout(feat)
        logits = self.head(feat)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result

AutoConfig.register("hf-sentiment-classifier", SentimentConfig)
AutoModel.register(SentimentConfig, SentimentClassifier)

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int = 128):
        self.texts = df["text"].astype(str).tolist()
        label_map = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
            "0": 0,
            "1": 1,
            "2": 2,
        }
        self.labels = df["label"].apply(
            lambda x: label_map.get(
                str(x).strip(),
                int(x) if str(x).strip().isdigit() else -1,
            )
        ).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_history(history, save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["val_acc"], "r-o", label="Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "val_accuracy_curve.png"))
    plt.close()

def train_solo_model():
    set_seed(SEED)

    checkpoint_dir = os.path.join(OUT_DIR, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    full = pd.read_csv(DATASET_PATH)
    full = full.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df, val_df = train_test_split(
        full,
        test_size=0.1,
        stratify=full["label"],
        random_state=SEED,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = SentimentDataset(train_df, tokenizer)
    val_ds = SentimentDataset(val_df, tokenizer)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    config = SentimentConfig(model_name=MODEL_NAME)
    model = SentimentClassifier(config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n or 1):.4f}")

        epoch_loss = running_loss / len(train_dl)
        history["train_loss"].append(epoch_loss)

        model.eval()
        all_preds, all_labels = [], []

        with torch.inference_mode():
            for batch in val_dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds = torch.argmax(outputs["logits"], dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        val_acc = accuracy_score(all_labels, all_preds)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
            pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2]).to_csv(
                os.path.join(OUT_DIR, "confusion_matrix.csv")
            )

            rpt = classification_report(
                all_labels,
                all_preds,
                digits=4,
                labels=[0, 1, 2],
                target_names=["Negative", "Neutral", "Positive"],
            )
            with open(
                os.path.join(OUT_DIR, "classification_report.txt"), "w"
            ) as f:
                f.write(rpt)

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix (Best Epoch)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
            plt.close()

    print(f"Best Acc: {best_acc:.4f}")

    plot_history(history, OUT_DIR)

    summary_data = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "best_val_accuracy": best_acc,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Path not found: {DATASET_PATH}")
    else:
        train_solo_model()
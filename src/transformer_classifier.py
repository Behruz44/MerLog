import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from preprocessing import clean_text, MERLOG_INTENTS

MODEL_NAME = "distilbert-base-uncased"
DEFAULT_SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "distilbert_intent",
)

LABEL2ID = {label: i for i, label in enumerate(MERLOG_INTENTS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # max_len 128 хотя тексты короткие и самые длинные где то 90 символов,
        # сначала пробовал 64 чтобы сэкономить память но на паре длинных жалоб
        # обрезало концовку и смысл терялся, 128 даёт запас а лишние токены
        # добиваются паддингом и почти не влияют на скорость
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_processed(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "processed",
            "intents_filtered.csv",
        )
    df = pd.read_csv(path)
    df["clean_instruction"] = df["clean_instruction"].fillna("").astype(str)
    return df


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def train_transformer(df, epochs=3, batch_size=16, test_size=0.2, save_dir=None):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    save_dir = save_dir or DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    X = df["clean_instruction"].tolist()
    y = [LABEL2ID[i] for i in df["intent"].tolist()]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(MERLOG_INTENTS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = IntentDataset(X_tr, y_tr, tokenizer)
    test_ds = IntentDataset(X_te, y_te, tokenizer)

    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    report = classification_report(
        [ID2LABEL[i] for i in y_te],
        [ID2LABEL[i] for i in y_pred],
        output_dict=True,
    )

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    metrics = {
        "accuracy": eval_result["eval_accuracy"],
        "f1_macro": eval_result["eval_f1_macro"],
        "report": report,
    }
    return model, tokenizer, metrics


def predict_transformer(text, model=None, tokenizer=None, save_dir=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    save_dir = save_dir or DEFAULT_SAVE_DIR
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(save_dir)

    cleaned = clean_text(text)
    enc = tokenizer(cleaned, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
    proba = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(proba))
    return ID2LABEL[idx], float(proba[idx])


if __name__ == "__main__":
    df = load_processed()
    print(f"loaded {len(df)} rows")
    model, tokenizer, metrics = train_transformer(df, epochs=3, batch_size=16)
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"f1_macro: {metrics['f1_macro']:.4f}")

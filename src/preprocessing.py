import re
import pandas as pd

MERLOG_INTENTS = [
    "track_order",
    "delivery_options",
    "complaint",
    "check_invoice",
    "cancel_order",
    "change_order",
]

WHITESPACE_RE = re.compile(r"\s+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")


def clean_text(text, keep_placeholders=True):
    # плейсхолдеры типа {{Order Number}} оставляю потому что в датасете
    # они почти в каждой фразе и по сути маркер что клиент упоминает
    # конкретный заказ, если выкинуть модель теряет этот сигнал
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    if not keep_placeholders:
        t = PLACEHOLDER_RE.sub("", t)
    t = NON_ASCII_RE.sub("", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text):
    t = clean_text(text)
    if not t:
        return []
    return t.split()


def filter_intents(df, intents=None):
    targets = intents or MERLOG_INTENTS
    sub = df[df["intent"].isin(targets)].copy()
    sub["clean_instruction"] = sub["instruction"].apply(clean_text)
    sub = sub[sub["clean_instruction"].str.len() > 0]
    return sub.reset_index(drop=True)


def load_raw(path):
    return pd.read_csv(path)


def prepare_dataset(raw_path, out_path, intents=None):
    df = load_raw(raw_path)
    filtered = filter_intents(df, intents)
    filtered.to_csv(out_path, index=False)
    return filtered


if __name__ == "__main__":
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(root, "data", "raw", "bitext_raw.csv")
    out = os.path.join(root, "data", "processed", "intents_filtered.csv")
    df = prepare_dataset(raw, out)
    print(f"saved {len(df)} rows to {out}")
    print(df.groupby("intent").size())
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import clean_text, MERLOG_INTENTS

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "intent_model.pkl",
)


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


def build_pipeline():
    # взял LogReg а не LinearSVM хотя обе дают похожий результат, но дальше
    # нужен confidence-роутер на predict_proba, а SVM вероятности из коробки
    # не отдаёт и пришлось бы оборачивать в CalibratedClassifierCV
    return Pipeline(
        [
            (
                "tfidf",
                # биграммы добавил потому что пары типа cancel order или
                # track shipment несут смысл который униграммы теряют,
                # без них cancel_order и change_order путались чаще
                TfidfVectorizer(
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                # C=2.0 чуть ослабил регуляризацию, датасет чистый и
                # переобучения не заметил, class_weight balanced на всякий
                # случай хотя классы почти ровные по тысяче на каждый
                LogisticRegression(
                    max_iter=1000,
                    C=2.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df["clean_instruction"].tolist()
    y = df["intent"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")
    report = classification_report(y_te, y_pred, output_dict=True)
    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "report": report,
        "labels": pipe.classes_.tolist(),
    }
    return pipe, metrics, (X_tr, X_te, y_tr, y_te)


def save_model(pipe: Pipeline, path: str | None = None):
    path = path or DEFAULT_MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    return path


def load_model(path: str | None = None) -> Pipeline:
    path = path or DEFAULT_MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(pipe: Pipeline, text: str) -> tuple[str, float]:
    cleaned = clean_text(text)
    proba = pipe.predict_proba([cleaned])[0]
    idx = int(np.argmax(proba))
    label = pipe.classes_[idx]
    conf = float(proba[idx])
    return label, conf


if __name__ == "__main__":
    df = load_processed()
    print(f"loaded {len(df)} rows")
    pipe, metrics, _ = train(df)
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"f1_macro: {metrics['f1_macro']:.4f}")
    path = save_model(pipe)
    print(f"model saved to {path}")
    for label in MERLOG_INTENTS:
        sample = df[df["intent"] == label]["clean_instruction"].iloc[0]
        pred, conf = predict(pipe, sample)
        print(f"  {label:20s} -> pred={pred:20s} conf={conf:.3f}")

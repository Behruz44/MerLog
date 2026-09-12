import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from entity_extractor import evaluate_extractor
from intent_classifier import load_model, load_processed, predict
from response_generator import generate_response
from router import route, evaluate_thresholds

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report")


def evaluate_on_bitext_split(pipe, df):
    # метрики на Bitext test split, тут фразы шаблонные и модель на них
    # показывает почти потолок точности, это ожидаемо и не значит что на
    # реальных обращениях будет так же
    X = df["clean_instruction"].tolist()
    y = df["intent"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_pred = [predict(pipe, t)[0] for t in X_te]
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")
    report = classification_report(y_te, y_pred, output_dict=True)
    labels = sorted(set(y))
    cm = confusion_matrix(y_te, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_test": len(y_te),
    }


def evaluate_on_synthetic(pipe):
    # метрики на synthetic наборе с реальными shipment ID, тут фразы
    # написаны руками и ближе к реальным обращениям, confidence ниже
    # потому что модель не видела таких формулировок в тренировке
    synthetic_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "synthetic_entities.json"
    )
    with open(synthetic_path, encoding="utf-8") as f:
        data = json.load(f)

    correct = 0
    auto = 0
    escalated = 0
    results = []
    for item in data:
        text = item["text"]
        expected = item["intent"]
        decision = route(text, pipe, response_fn=generate_response)
        is_correct = decision.intent == expected
        if decision.auto_respond:
            auto += 1
            if is_correct:
                correct += 1
        else:
            escalated += 1
        results.append({
            "text": text,
            "expected": expected,
            "predicted": decision.intent,
            "confidence": round(decision.confidence, 4),
            "shipment_id": decision.shipment_id,
            "auto_respond": decision.auto_respond,
            "correct": is_correct,
        })

    auto_accuracy = correct / auto if auto > 0 else 0
    escalation_rate = escalated / len(data)
    return {
        "n_total": len(data),
        "n_auto": auto,
        "n_escalated": escalated,
        "auto_accuracy": auto_accuracy,
        "escalation_rate": escalation_rate,
        "results": results,
    }


def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title="Confusion Matrix (Bitext test split)",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_curve(threshold_results, out_path):
    thresholds = [r["threshold"] for r in threshold_results]
    auto_rates = [r["automation_rate"] for r in threshold_results]
    error_rates = [r["error_rate"] for r in threshold_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, auto_rates, "o-", label="Automation rate", color="green")
    ax.plot(thresholds, error_rates, "s-", label="Error rate (among auto)", color="red")
    ax.set(xlabel="Confidence threshold", ylabel="Rate",
           title="Threshold vs Automation/Error rate (Bitext test split)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_full_evaluation():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_processed()
    pipe = load_model()

    # --- Bitext test split ---
    print("=== Bitext test split ===")
    bitext_metrics = evaluate_on_bitext_split(pipe, df)
    print(f"accuracy:  {bitext_metrics['accuracy']:.4f}")
    print(f"f1_macro: {bitext_metrics['f1_macro']:.4f}")
    print(f"n_test:   {bitext_metrics['n_test']}")
    print("(высокая точность ожидаема: датасет шаблонный, фразы внутри интента похожи)")

    cm = np.array(bitext_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, bitext_metrics["labels"],
                          os.path.join(REPORT_DIR, "confusion_matrix.png"))

    # --- Threshold analysis on Bitext split ---
    print("\n=== Threshold analysis (Bitext test split) ===")
    X = df["clean_instruction"].tolist()
    y = df["intent"].tolist()
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # взял широкий диапазон порогов от 0.5 до 0.95 потому что хотел увидеть
    # всю картину целиком а не только правильную точку, ниже 0.5 уже опасно
    # отдавать боту а выше 0.95 автоматизация падает так сильно что бот
    # становится почти бесполезным, этот график потом идёт в отчёт потому
    # что он хорошо показывает trade off между долей автоматических
    # ответов и долей ошибок среди них
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    thr_results = evaluate_thresholds(pipe, X_te, y_te, thresholds)
    for r in thr_results:
        print(f"  t={r['threshold']:.2f}  auto={r['automation_rate']:.3f}  "
              f"err={r['error_rate']:.3f}  ({r['auto_count']}/{len(X_te)})")
    plot_threshold_curve(thr_results, os.path.join(REPORT_DIR, "threshold_curve.png"))

    # --- Synthetic set ---
    print("\n=== Synthetic set (real shipment IDs) ===")
    syn_metrics = evaluate_on_synthetic(pipe)
    print(f"n_total:        {syn_metrics['n_total']}")
    print(f"n_auto:         {syn_metrics['n_auto']}")
    print(f"n_escalated:    {syn_metrics['n_escalated']}")
    print(f"auto_accuracy:  {syn_metrics['auto_accuracy']:.4f}")
    print(f"escalation_rate: {syn_metrics['escalation_rate']:.4f}")
    print("(confidence ниже потому что модель не видела реальные ID в тренировке)")
    for r in syn_metrics["results"]:
        status = "AUTO" if r["auto_respond"] else "ESCAL"
        ok = "OK" if r["correct"] else "MISS"
        print(f"  [{status}/{ok}] conf={r['confidence']:.3f} {r['text'][:40]:40s} "
              f"-> {r['predicted']} (expected {r['expected']})")

    # --- Entity extractor ---
    print("\n=== Entity extractor ===")
    ent_metrics = evaluate_extractor()
    print(f"precision: {ent_metrics['precision']:.4f}")
    print(f"recall:    {ent_metrics['recall']:.4f}")
    print(f"f1:        {ent_metrics['f1']:.4f}")
    print(f"(n={ent_metrics['total']}, это базовая проверка а не статистика)")

    # --- Save report ---
    full_report = {
        "bitext_test_split": {
            "accuracy": bitext_metrics["accuracy"],
            "f1_macro": bitext_metrics["f1_macro"],
            "n_test": bitext_metrics["n_test"],
            "report": bitext_metrics["report"],
            "note": "high accuracy expected: dataset is template-based, phrases within intent are similar",
        },
        "synthetic_set": {
            "n_total": syn_metrics["n_total"],
            "n_auto": syn_metrics["n_auto"],
            "n_escalated": syn_metrics["n_escalated"],
            "auto_accuracy": syn_metrics["auto_accuracy"],
            "escalation_rate": syn_metrics["escalation_rate"],
            "note": "confidence lower because model never saw real shipment IDs in training",
            "results": syn_metrics["results"],
        },
        "threshold_analysis": thr_results,
        "entity_extractor": {
            **ent_metrics,
            "note": "n=25, basic check not statistical proof, production needs larger set with varied formats",
        },
    }
    report_path = os.path.join(REPORT_DIR, "evaluation_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nreport saved to {report_path}")


if __name__ == "__main__":
    run_full_evaluation()

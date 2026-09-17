import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from entity_extractor import SYNTHETIC_PATH, evaluate_extractor
from intent_classifier import load_model, load_processed, predict
from response_generator import generate_response
from router import route, evaluate_thresholds

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report")


def bitext_test_split(df):
    # тот же split что и в intent_classifier.train, random_state=42
    _, X_te, _, y_te = train_test_split(
        df["clean_instruction"].tolist(), df["intent"].tolist(),
        test_size=0.2, stratify=df["intent"], random_state=42)
    return X_te, y_te


def evaluate_on_bitext_split(pipe, df):
    # метрики на Bitext test split, тут фразы шаблонные и модель на них
    # показывает почти потолок точности, это ожидаемо и не значит что на
    # реальных обращениях будет так же
    X_te, y_te = bitext_test_split(df)
    y_pred = [predict(pipe, t)[0] for t in X_te]
    labels = sorted(set(y_te))
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "f1_macro": f1_score(y_te, y_pred, average="macro"),
        "report": classification_report(y_te, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels),
        "labels": labels,
        "n_test": len(y_te),
    }


def evaluate_on_synthetic(pipe, use_keyword_fallback=True, threshold=0.5):
    # метрики на synthetic наборе с реальными shipment ID, тут фразы
    # написаны руками и ближе к реальным обращениям, confidence ниже
    # потому что модель не видела таких формулировок в тренировке,
    # threshold передаётся явно чтобы можно было проверить и на 0.5
    # (демо) и на 0.75 (продакшн)
    with open(SYNTHETIC_PATH, encoding="utf-8") as f:
        examples = json.load(f)

    rows = []
    for ex in examples:
        decision = route(ex["text"], pipe, threshold, generate_response, use_keyword_fallback)
        rows.append({
            "text": ex["text"],
            "expected": ex["intent"],
            "predicted": decision.intent,
            "confidence": round(decision.confidence, 4),
            "shipment_id": decision.shipment_id,
            "auto_respond": decision.auto_respond,
            "correct": decision.intent == ex["intent"],
        })

    auto_rows = [r for r in rows if r["auto_respond"]]
    misses = [{k: r[k] for k in ("text", "expected", "predicted", "confidence")}
              for r in auto_rows if not r["correct"]]
    n_auto = len(auto_rows)
    return {
        "threshold": threshold,
        "n_total": len(rows),
        "n_auto": n_auto,
        "n_escalated": len(rows) - n_auto,
        "auto_accuracy": (n_auto - len(misses)) / n_auto if n_auto else 0,
        "escalation_rate": (len(rows) - n_auto) / len(rows),
        "n_misclassifications": len(misses),
        "misclassifications": misses,
        "results": rows,
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


def print_threshold_rows(rows, n_test):
    for r in rows:
        print(f"  t={r['threshold']:.2f}  auto={r['automation_rate']:.3f}  "
              f"err={r['error_rate']:.3f}  ({r['auto_count']}/{n_test})")


def print_synthetic_summary(syn):
    print(f"n_auto: {syn['n_auto']}/{syn['n_total']}  auto_acc: {syn['auto_accuracy']:.4f}  "
          f"escal: {syn['n_escalated']}  miss: {syn['n_misclassifications']}")
    for m in syn["misclassifications"]:
        print(f"  MISS: '{m['text'][:50]}' -> {m['predicted']} (expected {m['expected']}, conf={m['confidence']})")


def run_full_evaluation():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_processed()
    pipe = load_model()

    print("=== Bitext test split ===")
    bitext = evaluate_on_bitext_split(pipe, df)
    print(f"accuracy:  {bitext['accuracy']:.4f}")
    print(f"f1_macro: {bitext['f1_macro']:.4f}")
    print(f"n_test:   {bitext['n_test']}")
    print("(высокая точность ожидаема: датасет шаблонный, фразы внутри интента похожи)")
    plot_confusion_matrix(bitext["confusion_matrix"], bitext["labels"],
                          os.path.join(REPORT_DIR, "confusion_matrix.png"))

    # threshold analysis на Bitext split
    print("\n=== Threshold analysis (Bitext test split) ===")
    X_te, y_te = bitext_test_split(df)
    # взял широкий диапазон порогов от 0.5 до 0.95 потому что хотел увидеть
    # всю картину целиком а не только правильную точку, ниже 0.5 уже опасно
    # отдавать боту а выше 0.95 автоматизация падает так сильно что бот
    # становится почти бесполезным, этот график потом идёт в отчёт потому
    # что он хорошо показывает trade off между долей автоматических
    # ответов и долей ошибок среди них
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # чистый ML без keyword подстраховки
    print("-- pure ML (no keyword fallback) --")
    thr_pure = evaluate_thresholds(pipe, X_te, y_te, thresholds, use_keyword_fallback=False)
    print_threshold_rows(thr_pure, len(X_te))

    # ML плюс keyword подстраховка
    print("-- ML + keyword fallback --")
    thr_kw = evaluate_thresholds(pipe, X_te, y_te, thresholds, use_keyword_fallback=True)
    print_threshold_rows(thr_kw, len(X_te))
    plot_threshold_curve(thr_kw, os.path.join(REPORT_DIR, "threshold_curve.png"))

    # synthetic set с реальными shipment ID
    print("\n=== Synthetic set (real shipment IDs) ===")
    print("-- pure ML, threshold=0.5 (demo) --")
    syn_pure = evaluate_on_synthetic(pipe, use_keyword_fallback=False, threshold=0.5)
    print_synthetic_summary(syn_pure)

    print("-- ML + keyword, threshold=0.5 (demo) --")
    syn_kw = evaluate_on_synthetic(pipe, use_keyword_fallback=True, threshold=0.5)
    print_synthetic_summary(syn_kw)

    # продакшн порог 0.75 на synthetic — отдельная проверка
    print("-- ML + keyword, threshold=0.75 (production) --")
    syn_prod = evaluate_on_synthetic(pipe, use_keyword_fallback=True, threshold=0.75)
    print_synthetic_summary(syn_prod)
    print("(продакшн порог 0.75 не спасает от confident-но-неверных предсказаний)")
    print("(confidence ниже потому что модель не видела реальные ID в тренировке)")
    for r in syn_kw["results"]:
        status = "AUTO" if r["auto_respond"] else "ESCAL"
        ok = "OK" if r["correct"] else "MISS"
        print(f"  [{status}/{ok}] conf={r['confidence']:.3f} {r['text'][:40]:40s} "
              f"-> {r['predicted']} (expected {r['expected']})")

    # entity extractor
    print("\n=== Entity extractor ===")
    ent = evaluate_extractor()
    print(f"precision: {ent['precision']:.4f}")
    print(f"recall:    {ent['recall']:.4f}")
    print(f"f1:        {ent['f1']:.4f}")
    print(f"(n={ent['total']}, это базовая проверка а не статистика)")

    # сохранение отчёта
    full_report = {
        "bitext_test_split": {
            **{k: bitext[k] for k in ("accuracy", "f1_macro", "n_test", "report")},
            "note": "high accuracy expected: dataset is template-based, phrases within intent are similar",
        },
        "synthetic_set_pure_ml": {
            **syn_pure,
            "note": "pure ML, no keyword fallback, confidence lower because model never saw real shipment IDs in training",
        },
        "synthetic_set_with_keyword_fallback": {
            **syn_kw,
            "note": "ML + keyword fallback, keyword rules catch phrases where model was unsure",
        },
        "synthetic_set_production_threshold_0.75": {
            **syn_prod,
            "note": "production threshold 0.75 on synthetic set, shows that confident-but-wrong predictions still pass, calibration from Bitext does not fully transfer to less templated phrasing",
        },
        "threshold_analysis_pure_ml": thr_pure,
        "threshold_analysis_with_keyword_fallback": thr_kw,
        "entity_extractor": {
            **ent,
            "note": "n=25, basic check not statistical proof, production needs larger set with varied formats",
        },
    }
    report_path = os.path.join(REPORT_DIR, "evaluation_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nreport saved to {report_path}")


if __name__ == "__main__":
    run_full_evaluation()

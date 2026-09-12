import json
import os
import re

SHIPMENT_ID_RE = re.compile(r"MRL-\d{4}-\d{4}", re.IGNORECASE)
# номер MerLog это MRL потом год потом четыре цифры, сначала думал взять
# spaCy для извлечения сущностей но для фиксированного формата регулярка
# работает точнее и быстрее, spaCy оставил на будущее для дат и портов
# и названий компаний в жалобах но в эту версию не стал вставлять


def extract_shipment_id(text: str) -> str | None:
    match = SHIPMENT_ID_RE.search(text)
    if match:
        return match.group(0).upper()
    return None


def extract_entities(text: str) -> dict:
    shipment_id = extract_shipment_id(text)
    entities = {"shipment_id": shipment_id}
    return entities


def evaluate_extractor(test_path: str | None = None) -> dict:
    if test_path is None:
        test_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "processed",
            "synthetic_entities.json",
        )
    with open(test_path, encoding="utf-8") as f:
        data = json.load(f)

    tp = fp = fn = 0
    for item in data:
        expected = item.get("shipment_id")
        predicted = extract_shipment_id(item["text"])
        if expected and predicted:
            if predicted == expected:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif expected and not predicted:
            fn += 1
        elif predicted and not expected:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total": len(data),
    }


if __name__ == "__main__":
    metrics = evaluate_extractor()
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} total={metrics['total']}")

    samples = [
        "where is my shipment MRL-2024-8831",
        "i want to cancel my order",
        "track MRL-2024-0042 please",
    ]
    for s in samples:
        print(f"  {s!r:50s} -> {extract_entities(s)}")

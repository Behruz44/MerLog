import json
import os
import re

SHIPMENT_ID_RE = re.compile(r"MRL-\d{4}-\d{4}", re.IGNORECASE)
# номер MerLog это MRL потом год потом четыре цифры, сначала думал взять
# spaCy для извлечения сущностей но для фиксированного формата регулярка
# работает точнее и быстрее, spaCy оставил на будущее для дат и портов
# и названий компаний в жалобах но в эту версию не стал вставлять

SYNTHETIC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "synthetic_entities.json",
)


def extract_shipment_id(text):
    m = SHIPMENT_ID_RE.search(text)
    return m.group(0).upper() if m else None


def extract_entities(text):
    return {"shipment_id": extract_shipment_id(text)}


def evaluate_extractor(test_path=SYNTHETIC_PATH):
    with open(test_path, encoding="utf-8") as f:
        examples = json.load(f)

    pairs = [(ex.get("shipment_id"), extract_shipment_id(ex["text"])) for ex in examples]
    tp = sum(1 for exp, found in pairs if exp and found == exp)
    # неверный или лишний ID считается как fp, пропущенный как fn
    fp = sum(1 for exp, found in pairs if found and found != exp)
    fn = sum(1 for exp, found in pairs if exp and found != exp)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "total": len(examples)}

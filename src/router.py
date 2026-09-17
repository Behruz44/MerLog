from dataclasses import dataclass

from entity_extractor import extract_entities
from intent_classifier import predict
from response_generator import detect_language


@dataclass
class RouterDecision:
    intent: str
    confidence: float
    shipment_id: str | None
    auto_respond: bool
    response: str
    raw_text: str


DEFAULT_THRESHOLD = 0.5
# порог для демо, бот отвечает чаще чтобы показать как работает,
# для продакшна лучше 0.75: 96.2% автоматизации на чистом ML и 0 ошибок,
# с keyword подстраховкой 97.8%, цифры в report/evaluation_results.json

# это демо-заглушка а не ML, классификатор обучен на английском
# и русский не понимает, для русского тут простой keyword matching с
# фиксированным confidence 0.8, для настоящего продакшна нужен либо
# переводчик перед классификатором либо отдельная модель на русском
RU_KEYWORDS = {
    "track_order": ["отслед", "где мой", "где груз", "статус", "когда приедет", "куда доехал", "где посылк", "где отправлен"],
    "delivery_options": ["вариант", "способ", "сколько идёт", "сколько идет", "доставка", "опции"],
    "complaint": ["жалоб", "претенз", "плох", "ужас", "поврежд", "битый", "сломал", "недовол", "пожалов"],
    "check_invoice": ["счёт", "счет", "инвойс", "чек", "документ", "накладн"],
    "cancel_order": ["отмен", "аннулир"],
    "change_order": ["измен", "поменять", "заменить", "обновить", "другой адрес", "поменять адрес"],
}

# английские keywords как подстраховка для фраз где модель путается
EN_KEYWORDS = {
    "track_order": ["where is my", "track my", "track order", "track shipment", "shipment status", "where is shipment"],
    "delivery_options": ["delivery option", "shipping method", "how long does", "delivery time"],
    "complaint": ["complain", "terrible", "damaged", "awful", "horrible", "worst"],
    "check_invoice": ["invoice", "bill", "receipt"],
    "cancel_order": ["cancel", "refund"],
    "change_order": ["change", "modify", "update"],
}


ESCALATE_MSG = {
    "ru": "Не уверен, что правильно понял запрос. Переключу вас на оператора MerLog.",
    "en": "I'm not confident I understood your request correctly. "
          "Let me connect you with a MerLog operator who can help.",
}


def keyword_match(text, keywords):
    # confidence условный, 0.8 чтобы пройти порог 0.5
    lowered = text.lower()
    hits = {intent: sum(kw in lowered for kw in kws) for intent, kws in keywords.items()}
    hits = {k: v for k, v in hits.items() if v}
    if not hits:
        return None, 0.0
    return max(hits, key=hits.get), 0.8


def route(text, pipe, threshold=DEFAULT_THRESHOLD, response_fn=None, use_keyword_fallback=True):
    lang = detect_language(text)
    shipment_id = extract_entities(text)["shipment_id"]

    if lang == "ru":
        intent, confidence = keyword_match(text, RU_KEYWORDS)
        if intent is None:
            intent, confidence = "unknown", 0.0
    else:
        intent, confidence = predict(pipe, text)
        # если модель неуверенна, пробуем keyword подстраховку
        if use_keyword_fallback and confidence < threshold:
            kw_intent, kw_conf = keyword_match(text, EN_KEYWORDS)
            if kw_intent is not None:
                intent, confidence = kw_intent, kw_conf

    auto = confidence >= threshold
    if not auto:
        response = ESCALATE_MSG[lang]
    elif response_fn:
        response = response_fn(intent, shipment_id, text)
    else:
        response = ""

    return RouterDecision(intent, confidence, shipment_id, auto, response, text)


def evaluate_thresholds(pipe, texts, true_intents, thresholds, response_fn=None, use_keyword_fallback=True):
    rows = []
    for thr in thresholds:
        auto_count = error_count = 0
        for text, true_intent in zip(texts, true_intents):
            decision = route(text, pipe, thr, response_fn, use_keyword_fallback)
            if not decision.auto_respond:
                continue
            auto_count += 1
            error_count += decision.intent != true_intent
        rows.append({
            "threshold": thr,
            "automation_rate": auto_count / len(texts),
            "error_rate": error_count / auto_count if auto_count else 0,
            "auto_count": auto_count,
            "error_count": error_count,
        })
    return rows

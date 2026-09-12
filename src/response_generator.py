import re

from entity_extractor import extract_shipment_id


# шаблоны вместо генерации, сначала думал прикрутить что то вроде Claude
# для ответов но для статус бота это создаёт больше проблем чем решает,
# модель может выдумать статус или назвать дату доставки которой нет,
# а в логистике это прямая жалоба, шаблоны дают предсказуемый ответ и
# для MerLog надёжность важнее чем живость формулировок

TEMPLATES = {
    "track_order": {
        "en": {
            "with_id": "Your shipment {shipment_id} is currently in transit. "
            "You can check the latest status in your MerLog account or contact an operator for details.",
            "without_id": "I'd be happy to help track your shipment. "
            "Could you please provide your shipment ID? It looks like MRL-YYYY-NNNN.",
        },
        "ru": {
            "with_id": "Ваш груз {shipment_id} сейчас в пути. "
            "Актуальный статус можно посмотреть в личном кабинете MerLog или уточнить у оператора.",
            "without_id": "Помогу отследить ваш груз. "
            "Назовите номер отправления, формат MRL-YYYY-NNNN.",
        },
    },
    "delivery_options": {
        "en": {
            "with_id": "For shipment {shipment_id}, available delivery options include standard (3-5 days), "
            "express (1-2 days), and same-day courier in select cities.",
            "without_id": "MerLog offers standard (3-5 days), express (1-2 days), "
            "and same-day courier in select cities. Which would you prefer?",
        },
        "ru": {
            "with_id": "Для груза {shipment_id} доступны: стандартная доставка (3-5 дней), "
            "экспресс (1-2 дня) и курьер в день обращения в крупных городах.",
            "without_id": "MerLog предлагает стандартную доставку (3-5 дней), экспресс (1-2 дня) "
            "и курьера в день обращения в крупных городах. Что выберете?",
        },
    },
    "complaint": {
        "en": {
            "with_id": "I'm sorry to hear about the issue with shipment {shipment_id}. "
            "Your complaint has been logged. A MerLog representative will follow up within 24 hours.",
            "without_id": "I'm sorry to hear you've had a negative experience. "
            "Could you share your shipment ID so I can log this complaint properly?",
        },
        "ru": {
            "with_id": "Жаль, что возникла проблема с грузом {shipment_id}. "
            "Жалоба зарегистрирована, специалист MerLog свяжется с вами в течение 24 часов.",
            "without_id": "Жаль слышать о неприятной ситуации. "
            "Назовите номер отправления, чтобы я мог оформить жалобу.",
        },
    },
    "check_invoice": {
        "en": {
            "with_id": "Your invoice for shipment {shipment_id} is available in your MerLog account "
            "under the Billing section. Would you like it resent to your email?",
            "without_id": "I can help you find your invoice. "
            "Could you provide the shipment ID associated with the invoice?",
        },
        "ru": {
            "with_id": "Счёт по грузу {shipment_id} доступен в личном кабинете MerLog "
            "в разделе Биллинг. Прислать копию на почту?",
            "without_id": "Помогу найти счёт. "
            "Назовите номер отправления, по которому нужен счёт.",
        },
    },
    "cancel_order": {
        "en": {
            "with_id": "I've initiated a cancellation request for shipment {shipment_id}. "
            "If it hasn't shipped yet, the cancellation will be processed within 1 hour.",
            "without_id": "I can help cancel your order. "
            "Please provide your shipment ID so I can locate the correct order.",
        },
        "ru": {
            "with_id": "Оформил заявку на отмену груза {shipment_id}. "
            "Если груз ещё не отправлен, отмена пройдёт в течение часа.",
            "without_id": "Помогу отменить заказ. "
            "Назовите номер отправления, чтобы я нашёл нужный заказ.",
        },
    },
    "change_order": {
        "en": {
            "with_id": "I've submitted a modification request for shipment {shipment_id}. "
            "Changes are possible if the shipment hasn't been dispatched yet.",
            "without_id": "I can help modify your order. "
            "Could you provide your shipment ID so I can find the right order?",
        },
        "ru": {
            "with_id": "Оформил заявку на изменение груза {shipment_id}. "
            "Изменения возможны, если груз ещё не отправлен.",
            "without_id": "Помогу изменить заказ. "
            "Назовите номер отправления, чтобы я нашёл нужный заказ.",
        },
    },
}

ESCALATION_MESSAGE = {
    "en": "I'm not confident I understood your request correctly. "
    "Let me connect you with a MerLog operator who can help.",
    "ru": "Не уверен, что правильно понял запрос. "
    "Переключу вас на оператора MerLog, он поможет.",
}

CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")


def detect_language(text):
    # если в тексте есть кириллица, считаем что русский
    if CYRILLIC_RE.search(text):
        return "ru"
    return "en"


def generate_response(intent, shipment_id, text=""):
    lang = detect_language(text)
    intent_templates = TEMPLATES.get(intent)
    if intent_templates is None:
        return ESCALATION_MESSAGE[lang]
    lang_templates = intent_templates.get(lang, intent_templates["en"])
    if shipment_id:
        return lang_templates["with_id"].format(shipment_id=shipment_id)
    return lang_templates["without_id"]


if __name__ == "__main__":
    for intent in TEMPLATES:
        print(f"[{intent}] en with_id:")
        print(f"  {generate_response(intent, 'MRL-2024-8831', 'where is my shipment')}")
        print(f"[{intent}] ru with_id:")
        print(f"  {generate_response(intent, 'MRL-2024-8831', 'где мой груз')}")
        print(f"[{intent}] en without_id:")
        print(f"  {generate_response(intent, None, 'track my order')}")
        print(f"[{intent}] ru without_id:")
        print(f"  {generate_response(intent, None, 'отследить заказ')}")
        print()

# MerLog Chatbot

chatbot for tracking shipment status and customer support, built for MerLog logistics

## What it does

the bot takes a customer message, classifies the intent (track, complain, cancel, etc), extracts the shipment ID if present, and if the model is confident enough it responds automatically using a template, otherwise it escalates to a human operator

```
customer message
    │
    ▼
[Preprocessing] ── text cleaning
    │
    ▼
[Intent Classifier] ── intent + confidence
    │
    ▼
[Entity Extractor] ── shipment ID (MRL-YYYY-NNNN)
    │
    ▼
[Confidence Router] ── confidence < threshold? ──► human operator
    │ no
    ▼
[Response Generator] ── template-based reply
```

## Dataset

took the Bitext Customer Support Training Dataset from HuggingFace, it has 26872 examples across 27 intents, but MerLog is logistics and not all of them are needed, so I filtered down to six: track_order, delivery_options, complaint, check_invoice, cancel_order, change_order, ended up with 5985 examples

the rest like create_account and newsletter_subscription are not relevant to MerLog and would just add noise to the model

I also wrote 25 examples myself with realistic MerLog customer phrases that contain shipment IDs, this is needed because the main dataset has no real IDs, only placeholders like {{Order Number}}

dataset: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

## Models

trained two variants to compare:

| Model | Accuracy | F1 macro | Train time | Inference |
|---|---|---|---|---|
| TF-IDF + LogReg | 99.58% | 0.9958 | ~2s | <1 ms |
| DistilBERT | 99.75% | 0.9975 | ~2 min | ~15 ms |

chose TF-IDF plus LogReg for production because the accuracy gap is only 0.17% but inference is 15x faster, for a real-time chatbot this is critical

## Results

two separate test sets:

**Bitext test split** (1197 examples, template phrases):
- classifier: 99.58% accuracy, 0.9958 F1 macro
- pure ML at threshold 0.75: 96.2% automation, 0% errors
- ML + keyword fallback at threshold 0.75: 97.8% automation, 0% errors
- keyword fallback adds ~1.6% automation by catching phrases where the model was right but unsure
- high accuracy is expected because the dataset is template-based, phrases within an intent are structurally similar, this does not guarantee the same on real customer messages

**Synthetic set** (25 examples with real shipment IDs):
- 11 auto responses (44%), 14 escalations (56%)
- auto accuracy: 90.9% (9 out of 11 correct)
- 1 misclassification: "change delivery address" classified as delivery_options instead of change_order
- confidence is lower because the model never saw real shipment IDs in training, only {{Order Number}} placeholders

**Entity extractor**:
- 100% precision and recall on 25 examples
- this is a basic check not statistical proof, production needs a larger set with varied formats (no hyphens, two-digit years, etc)

## Russian language support

the classifier is trained on English Bitext data and does not understand Russian, for the demo I added a keyword-matching fallback in `router.py` that detects Russian text and maps it to intents via hardcoded keyword lists, confidence is fixed at 0.8 (not a real probability), this is a demo stub not a production feature, for real multilingual support a translator or a separate Russian model would be needed

the same keyword fallback also runs on English when the ML model is unsure, this is a hybrid ML + rule-based approach, the threshold analysis in the report shows both pure ML and ML + keyword numbers separately

## Thresholds

- demo (app.py, CLI, Streamlit): threshold 0.5, bot responds more often for demonstration
- production recommendation: threshold 0.75, 96.2% automation with pure ML and 0% errors, or 97.8% with keyword fallback

## Setup

```bash
pip install -r requirements.txt
```

## Usage

prepare data:

```bash
python src/preprocessing.py
```

downloads the Bitext dataset to data/raw/ and saves filtered intents to data/processed/intents_filtered.csv

train classifier:

```bash
python src/intent_classifier.py
```

trains TF-IDF plus LogReg, saves model to data/processed/intent_model.pkl

train transformer (optional, for comparison):

```bash
python src/transformer_classifier.py
```

run evaluation:

```bash
python src/evaluate.py
```

generates confusion matrix, threshold curve, and end-to-end demo in report/

CLI chatbot:

```bash
python app.py
```

Streamlit demo:

```bash
streamlit run app.py -- --mode streamlit
```

## Shipment ID format

MerLog shipment ID: MRL then year then four digits, for example MRL-2024-8831, extracted via regex

## Project structure

```
merlog-chatbot/
├── README.md
├── app.py                        # CLI / Streamlit
├── requirements.txt
├── data/
│   ├── raw/                      # Bitext dataset (not committed)
│   └── processed/                # filtered intents, synthetic, model
├── src/
│   ├── preprocessing.py          # text cleaning, intent filtering
│   ├── intent_classifier.py      # TF-IDF + LogReg
│   ├── transformer_classifier.py # DistilBERT (comparison)
│   ├── entity_extractor.py       # shipment ID extraction
│   ├── router.py                 # confidence-threshold routing
│   ├── response_generator.py     # template-based responses
│   └── evaluate.py               # metrics, plots, demo
├── notebooks/
│   └── experiments.ipynb         # model comparison
└── report/                       # evaluation results, plots
```

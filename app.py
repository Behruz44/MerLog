import argparse
import sys

from entity_extractor import extract_entities
from intent_classifier import load_model, predict
from response_generator import generate_response
from router import DEFAULT_THRESHOLD, route


def run_cli(threshold=DEFAULT_THRESHOLD):
    pipe = load_model()
    print("MerLog Chatbot (CLI). Type 'quit' to exit.")
    print("-" * 50)
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            continue
        decision = route(text, pipe, threshold=threshold, response_fn=generate_response)
        status = "AUTO" if decision.auto_respond else "ESCALATE"
        print(f"[{status}] intent={decision.intent} conf={decision.confidence:.3f} id={decision.shipment_id}")
        print(f"Bot: {decision.response}")
        print()


def run_streamlit(threshold=DEFAULT_THRESHOLD):
    import streamlit as st

    st.set_page_config(page_title="MerLog Chatbot", page_icon=":package:")
    st.title("MerLog Shipment Chatbot")
    st.caption("Track shipments, file complaints, manage orders.")

    pipe = load_model()

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask about your shipment...")
    if user_input:
        decision = route(user_input, pipe, threshold=threshold, response_fn=generate_response)
        st.session_state.history.append({"role": "user", "text": user_input})
        st.session_state.history.append(
            {
                "role": "bot",
                "text": decision.response,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "shipment_id": decision.shipment_id,
                "auto": decision.auto_respond,
            }
        )

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["text"])
                if not msg["auto"]:
                    st.caption("Escalated to human operator")
                else:
                    st.caption(
                        f"intent: {msg['intent']} | confidence: {msg['confidence']:.1%}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MerLog Chatbot")
    parser.add_argument("--mode", choices=["cli", "streamlit"], default="cli")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if args.mode == "streamlit":
        run_streamlit(args.threshold)
    else:
        run_cli(args.threshold)

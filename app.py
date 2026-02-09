# app.py
from __future__ import annotations

import gradio as gr
from gradio_client import utils as gradio_client_utils

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from data import QA_DATA
from llm_fallback import generic_fallback, llm_fallback
from retrieval import FAQRetriever

if load_dotenv:
    load_dotenv()

retriever = FAQRetriever(QA_DATA)

_original_get_type = gradio_client_utils.get_type
_original_json_schema_to_python_type = gradio_client_utils.json_schema_to_python_type
_original_json_schema_to_python_type_inner = gradio_client_utils._json_schema_to_python_type


def _safe_get_type(schema: object):
    # gradio_client 1.3.0 can receive boolean JSON schema values; guard to avoid TypeError.
    if isinstance(schema, bool):
        return {}
    return _original_get_type(schema)


def _safe_json_schema_to_python_type(schema: object) -> str:
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema)


def _safe_json_schema_to_python_type_inner(schema: object, defs):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type_inner(schema, defs)


gradio_client_utils.get_type = _safe_get_type
gradio_client_utils.json_schema_to_python_type = _safe_json_schema_to_python_type
gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type_inner


def format_answer(answer: str, matched_q: str | None = None, score: float | None = None) -> str:
    lines: list[str] = []
    lines.append(answer)

    if matched_q is not None and score is not None:
        lines.append("")
        lines.append(f"Matched FAQ: “{matched_q}”")
        lines.append(f"Confidence: {score:.2f}")

    return "\n".join(lines)


def chatbot(user_message: str, history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    user_message = (user_message or "").strip()

    if not user_message:
        bot = "Please type a question (e.g., “What does EVA do?”)."
        history = history + [("", bot)]
        return history, ""

    result, confident = retriever.answer(user_message, threshold=0.22)

    if result and confident:
        bot = format_answer(result.answer, matched_q=result.question, score=result.score)
        history = history + [(user_message, bot)]
        return history, ""

    # Not confident: try LLM fallback if configured; otherwise use generic fallback.
    llm_text = llm_fallback(user_message)
    bot = llm_text if llm_text else generic_fallback(user_message)

    # Provide the closest FAQ hint for transparency.
    if result:
        bot = format_answer(bot, matched_q=result.question, score=result.score)

    history = history + [(user_message, bot)]
    return history, ""


with gr.Blocks(title="Thoughtful AI Support Agent") as demo:
    gr.Markdown(
        "# Thoughtful AI Support Agent\n"
        "Ask about EVA, CAM, PHIL, Thoughtful AI’s agents, or benefits. "
        "For anything else, the agent will fall back to a general response (or an LLM if configured)."
    )

    chatbot_ui = gr.Chatbot(label="Chat", height=420)
    msg = gr.Textbox(label="Your question", placeholder="e.g., What does EVA do?", autofocus=True)
    clear = gr.Button("Clear")

    msg.submit(chatbot, [msg, chatbot_ui], [chatbot_ui, msg])
    clear.click(lambda: [], None, chatbot_ui)

if __name__ == "__main__":
    try:
        demo.launch()
    except ValueError as exc:
        if "localhost is not accessible" in str(exc):
            demo.launch(share=True)
        else:
            raise

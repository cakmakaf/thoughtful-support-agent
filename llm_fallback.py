# llm_fallback.py
from __future__ import annotations

import os
from typing import Optional


def generic_fallback(_: str) -> str:
    # Works with zero external dependencies.
    return (
        "I may not have that specific detail in my FAQ set. "
        "If your question is about Thoughtful AI’s agents (EVA, CAM, PHIL) or their benefits, "
        "try asking using those names. Otherwise, share a bit more context and I’ll do my best to help."
    )


def llm_fallback(user_text: str) -> Optional[str]:
    """
    Optional: If you set OPENAI_API_KEY, you can get true LLM fallback.
    If no key is present (or the SDK isn't installed), returns None.

    Environment variables:
      - OPENAI_API_KEY (required for LLM fallback)
      - OPENAI_MODEL (optional; default: gpt-4o-mini)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # Lazy import so the project still runs without this dependency.
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a helpful customer support agent for Thoughtful AI. "
        "Answer the user clearly and concisely. If you are unsure, say what you need to know.\n\n"
        f"User: {user_text}"
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

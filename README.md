# Thoughtful AI Support Agent (Hardcoded FAQ + Fallback)

A minimal customer support agent that:
- Retrieves the most relevant answer from a hardcoded FAQ dataset (TF-IDF + cosine similarity)
- Falls back to a generic response for everything else
- Optionally uses an LLM for fallback if `OPENAI_API_KEY` is set

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the local URL printed in your terminal.

## Optional: LLM fallback

Set env vars (optional):

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"
python app.py
```

If you don't set `OPENAI_API_KEY`, the app will still run and use a generic fallback response.

## How retrieval works

We vectorize the FAQ questions using TF-IDF (unigrams + bigrams) and pick the closest question by cosine similarity.
If the similarity score is below a threshold, we treat it as low confidence and use fallback.

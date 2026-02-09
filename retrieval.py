# retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    question: str
    answer: str
    score: float


class FAQRetriever:
    """
    Simple semantic-ish retrieval using TF-IDF over the fixed FAQ questions.
    Great for small datasets and fast to run without external services.
    """

    def __init__(self, qa_pairs: List[dict]) -> None:
        if not qa_pairs:
            raise ValueError("qa_pairs must not be empty")

        self.qa_pairs = qa_pairs
        self.questions = [q["question"] for q in qa_pairs]

        # ngrams help phrase matching; stop_words reduces noise.
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.question_matrix = self.vectorizer.fit_transform(self.questions)

    def best_match(self, user_text: str) -> Optional[RetrievalResult]:
        user_text = (user_text or "").strip()
        if not user_text:
            return None

        user_vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, self.question_matrix)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        best_pair = self.qa_pairs[best_idx]

        return RetrievalResult(
            question=best_pair["question"],
            answer=best_pair["answer"],
            score=best_score,
        )

    def answer(self, user_text: str, threshold: float = 0.22) -> Tuple[Optional[RetrievalResult], bool]:
        """
        Returns (result, is_confident).
        For small FAQ sets, 0.20–0.30 is a reasonable starting range.
        """
        result = self.best_match(user_text)
        if result is None:
            return None, False
        return result, (result.score >= threshold)

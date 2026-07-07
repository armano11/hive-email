"""RAG (Retrieval-Augmented Generation) system for grounded reply generation.

Embeds past customer support examples and retrieves the most similar
cases for few-shot prompting. This grounds the LLM generation in
real (synthetic) past resolutions rather than relying on parametric
memory alone.

Design:
- Uses sentence-transformers for embedding
- Stores embeddings in a simple in-memory index (no FAISS dependency)
- Retrieves top-k similar cases by cosine similarity
- Falls back gracefully if embeddings unavailable

Tradeoff note: FAISS or Annoy would be needed at scale (>10K examples).
For our dataset size (~120 examples), brute-force cosine is fine.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Example:
    """A single retrieval example with metadata."""

    def __init__(
        self,
        email_text: str,
        gold_reply: str,
        intent: str,
        tone: str,
        urgency: str,
        expected_actions: list[str] | None = None,
    ):
        self.email_text = email_text
        self.gold_reply = gold_reply
        self.intent = intent
        self.tone = tone
        self.urgency = urgency
        self.expected_actions = expected_actions or []
        self.embedding: np.ndarray | None = None


class Retriever:
    """Simple embedding-based retriever for few-shot examples."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.examples: list[Example] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        except Exception as e:
            logger.warning("Failed to load sentence-transformers model: %s", e)
            self._model = None
        return self._model

    def index(self, examples: list[Example]) -> None:
        """Index a list of examples for retrieval."""
        self.examples = examples

        model = self._load_model()
        if model is None:
            logger.warning("No embedding model — retrieval will use random selection")
            self.embeddings = None
            return

        texts = [ex.gold_reply for ex in examples]
        self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        logger.info("Indexed %d examples with %d-dim embeddings", len(examples), self.embeddings.shape[1])

    def retrieve(
        self,
        email_text: str,
        intent: str | None = None,
        top_k: int = 3,
        same_intent_only: bool = True,
    ) -> list[tuple[Example, float]]:
        """Retrieve top-k most similar examples for few-shot prompting.

        Args:
            email_text: The customer email to find similar cases for
            intent: Filter by intent type
            top_k: Number of examples to return
            same_intent_only: If True, only return examples of the same intent

        Returns:
            List of (Example, similarity_score) tuples
        """
        model = self._load_model()
        if model is None or self.embeddings is None:
            return self._fallback_retrieve(intent, top_k)

        candidates = self.examples
        candidate_embs = self.embeddings

        if same_intent_only and intent:
            indices = [i for i, ex in enumerate(candidates) if ex.intent == intent]
            if not indices:
                indices = list(range(len(candidates)))
            candidates = [candidates[i] for i in indices]
            candidate_embs = self.embeddings[indices]

        if not candidates:
            return self._fallback_retrieve(intent, top_k)

        query_emb = model.encode(email_text, normalize_embeddings=True)

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_emb], candidate_embs)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append((candidates[idx], float(similarities[idx])))

        if not results:
            return self._fallback_retrieve(intent, top_k)

        logger.debug("Retrieved %d examples for intent=%s", len(results), intent)
        return results

    def _fallback_retrieve(
        self, intent: str | None = None, top_k: int = 3
    ) -> list[tuple[Example, float]]:
        """Fallback: return random examples of same intent."""
        if intent:
            matching = [ex for ex in self.examples if ex.intent == intent]
        else:
            matching = self.examples

        if not matching:
            matching = self.examples

        import random
        selected = random.sample(matching, min(top_k, len(matching)))
        return [(ex, 0.0) for ex in selected]


def build_retriever_from_dataset(
    dataset_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Retriever:
    """Build a retriever from a dataset JSON file."""
    import json

    with open(dataset_path) as f:
        data = json.load(f)

    retriever = Retriever(model_name=model_name)
    examples = []

    for item in data:
        ex = Example(
            email_text=item["customer_email"],
            gold_reply=item["gold_reply"],
            intent=item.get("intent", "unknown"),
            tone=item.get("tone", "neutral"),
            urgency=item.get("urgency", "medium"),
            expected_actions=item.get("expected_actions", []),
        )
        examples.append(ex)

    retriever.index(examples)
    return retriever

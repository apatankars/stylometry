"""Cross-encoder reranker head stub.

Reference: Nogueira & Cho "Passage Re-ranking with BERT", arXiv:1901.04085.

The cross-encoder takes (query_email, profile_email) pairs and produces a
scalar similarity score without the asymmetric limitations of bi-encoders.
Used as a second-stage reranker after the prototypical head shortlists
candidates.

Two-stage pipeline
------------------
Stage 1 — PrototypicalHead:  Fast O(1) centroid lookup.  Filters candidates
           to a shortlist (e.g. "this email looks anomalous").
Stage 2 — CrossEncoderHead:  Expensive joint encoding of (query, profile)
           pairs.  Only runs on the shortlist, keeping latency manageable.

Why the asymmetry matters
--------------------------
A bi-encoder (like HFEncoder) encodes query and profile independently; the
similarity is computed post-hoc.  This is fast but can miss fine-grained
cross-attention signals between the two texts.
A cross-encoder concatenates both texts and lets attention flow between them,
which is more accurate but ~10–100× slower.
"""

from __future__ import annotations

import torch

from email_fraud.heads.base import BaseHead

# @register("head", "cross_encoder") — Nogueira & Cho arXiv:1901.04085
from email_fraud.registry import register


@register("head", "cross_encoder")
class CrossEncoderHead(BaseHead):
    """Cross-encoder reranker for second-stage sender verification.

    Encodes (query, candidate) pairs jointly so the model can attend across
    both texts.  More accurate than the prototypical head but ~10× slower;
    intended for reranking a shortlist, not full-corpus scoring.

    Current status: NOT IMPLEMENTED.  The class is registered and the
    interface is defined so it can be wired into the pipeline config without
    changing other code.  The implementation is deferred until the
    prototypical head's performance ceiling is measured.
    """

    def __init__(self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        # TODO: load cross-encoder model via sentence-transformers or
        #       transformers AutoModelForSequenceClassification
        # NOTE: unlike the prototypical head which stores embedding tensors,
        # the cross-encoder needs the raw text so it can jointly encode pairs.
        # fit() must receive raw texts, not pre-computed embeddings — this
        # requires a different training loop path.
        self._profiles: dict[str, list[str]] = {}  # sid -> list of profile texts

    def fit(
        self,
        embeddings: torch.Tensor,
        sender_ids: list[str],
    ) -> None:
        # The cross-encoder cannot build profiles from embeddings because it
        # needs the original text to form (query, profile_text) pairs.
        # The training loop would need to pass texts directly instead.
        raise NotImplementedError(
            "CrossEncoderHead.fit() is not yet implemented. "
            "This head stores raw texts, not embeddings — override the training "
            "loop to pass texts directly."
        )

    def score(
        self,
        query: torch.Tensor,
        sender_id: str,
    ) -> dict[str, object]:
        # Would: retrieve stored profile texts for sender_id, encode each
        # (query_text, profile_text) pair through the cross-encoder, return
        # the max or mean score across profile texts.
        raise NotImplementedError(
            "CrossEncoderHead.score() is not yet implemented. "
            "TODO: run (query_text, profile_text) pairs through the cross-encoder "
            "and return the max / mean score."
        )

    def save(self, path: str) -> None:
        raise NotImplementedError("CrossEncoderHead.save() not yet implemented.")

    def load(self, path: str) -> None:
        raise NotImplementedError("CrossEncoderHead.load() not yet implemented.")

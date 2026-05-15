"""End-to-end inference: raw email → anomaly score.

Wires encoder → head → store. Pass update_on_score=True for online EWMA profile
updates; leave False for forensics/auditing where the profile must stay frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from email_fraud.config import ExperimentConfig, PreprocessingConfig
from email_fraud.data.preprocessing import preprocess
from email_fraud.encoders.base import BaseEncoder
from email_fraud.heads.base import BaseHead
from email_fraud.profiles.store import SenderProfileStore


@dataclass
class ScoringResult:
    """score is in [0, 1] (higher = more consistent with profile).
    abstain=True when the profile is too sparse to trust the score.
    """

    sender_id: str
    score: float
    tier: str
    abstain: bool
    embedding: torch.Tensor
    raw: dict[str, Any]


class ScoringPipeline:
    """Connects encoder → head → anomaly score for a claimed sender."""

    def __init__(
        self,
        encoder: BaseEncoder,
        head: BaseHead,
        store: SenderProfileStore,
        preprocessing: PreprocessingConfig | None = None,
        device: str = "cpu",
        update_on_score: bool = False,
    ) -> None:
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.head = head
        self.store = store
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.device = device
        self.update_on_score = update_on_score

    @torch.no_grad()
    def score(self, email_text: str, claimed_sender: str) -> ScoringResult:
        """Score a single email against the claimed sender's profile."""
        # Fall back to "" if preprocess returns None (email too short after cleaning).
        cleaned = preprocess(email_text, self.preprocessing) or ""
        token_dict = self.encoder.tokenize([cleaned])
        token_dict = {k: v.to(self.device) for k, v in token_dict.items()}

        embedding = self.encoder.encode(**token_dict)  # (1, d)
        query = embedding.squeeze(0).cpu()             # (d,)

        raw = self.head.score(query, claimed_sender)

        if self.update_on_score:
            self.store.upsert(claimed_sender, query.numpy())

        return ScoringResult(
            sender_id=claimed_sender,
            score=float(raw["score"]),
            tier=str(raw["tier"]),
            abstain=bool(raw["abstain"]),
            embedding=query,
            raw=raw,
        )

    @torch.no_grad()
    def score_batch(
        self,
        email_texts: list[str],
        claimed_senders: list[str],
    ) -> list[ScoringResult]:
        """Score a batch of (email, claimed_sender) pairs, encoding texts together."""
        if len(email_texts) != len(claimed_senders):
            raise ValueError(
                "email_texts and claimed_senders must have the same length."
            )

        cleaned = [preprocess(t, self.preprocessing) or "" for t in email_texts]
        token_dict = self.encoder.tokenize(cleaned)
        token_dict = {k: v.to(self.device) for k, v in token_dict.items()}

        embeddings = self.encoder.encode(**token_dict)  # (N, d)
        embeddings_cpu = embeddings.cpu()

        results = []
        for i, (emb, sid) in enumerate(zip(embeddings_cpu, claimed_senders)):
            raw = self.head.score(emb, sid)
            if self.update_on_score:
                self.store.upsert(sid, emb.numpy())
            results.append(
                ScoringResult(
                    sender_id=sid,
                    score=float(raw["score"]),
                    tier=str(raw["tier"]),
                    abstain=bool(raw["abstain"]),
                    embedding=emb,
                    raw=raw,
                )
            )
        return results

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        encoder: BaseEncoder,
        head: BaseHead,
        store: SenderProfileStore,
        device: str = "cpu",
    ) -> "ScoringPipeline":
        return cls(
            encoder=encoder,
            head=head,
            store=store,
            preprocessing=config.data.preprocessing,
            device=device,
        )

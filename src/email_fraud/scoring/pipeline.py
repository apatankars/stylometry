"""End-to-end inference pipeline: raw email → anomaly score.

The pipeline connects the three main components:
  encoder  →  embeds the query email
  head     →  compares embedding against stored sender profile
  store    →  provides the profile (optionally updates it post-decision)

It is intentionally stateless: pass encoder, head, and store as constructor
arguments so the pipeline can be swapped or unit-tested in isolation.

Online update mode
------------------
When update_on_score=True, each scored email is also upserted into the store
(EWMA update).  This allows the profile to adapt to a sender's evolving style
over time without a separate re-profiling step.  Disable for auditing / forensics
where you want the profile frozen at a known point in time.
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
    """Result returned by ScoringPipeline.score().

    Attributes:
        sender_id:    The claimed sender.
        score:        Float in [0, 1] — higher = more consistent with profile.
        tier:         Confidence tier string (low / medium / high / very_high).
        abstain:      True when the profile is too sparse to trust the score.
        embedding:    (d,) tensor of the query's embedding (detached, CPU).
        raw:          Full dict returned by the head (for logging / debugging).
    """

    sender_id: str
    score: float
    tier: str
    abstain: bool
    embedding: torch.Tensor
    raw: dict[str, Any]


class ScoringPipeline:
    """Connects encoder → head → anomaly score for a claimed sender.

    Args:
        encoder:          Trained BaseEncoder (in eval mode).
        head:             Fitted BaseHead with per-sender profiles loaded.
        store:            SenderProfileStore (used for tier lookup).
        preprocessing:    PreprocessingConfig controlling text cleaning.
        device:           Torch device string ("cpu", "cuda", etc.).
        update_on_score:  If True, upsert the query embedding into the store
                          after scoring (online profile update).
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, email_text: str, claimed_sender: str) -> ScoringResult:
        """Score a single email against the claimed sender's profile.

        Args:
            email_text:      Raw email body string.
            claimed_sender:  Sender id to compare against.

        Returns:
            ScoringResult with score, tier, abstain flag, and embedding.
        """
        # Preprocess handles reply/sig stripping and entity masking.
        # None is returned if the email is too short after cleaning;
        # fall back to empty string so we still get a (low-confidence) embedding.
        cleaned = preprocess(email_text, self.preprocessing) or ""
        token_dict = self.encoder.tokenize([cleaned])
        token_dict = {k: v.to(self.device) for k, v in token_dict.items()}

        embedding = self.encoder.encode(**token_dict)  # (1, d)
        query = embedding.squeeze(0).cpu()             # (d,)

        raw = self.head.score(query, claimed_sender)

        # Optionally update the sender's profile with the newly scored email.
        # This is the "online learning" path for production systems.
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
        """Score a batch of (email, claimed_sender) pairs.

        Each email is scored independently against its claimed sender; texts
        are encoded together for efficiency.
        """
        if len(email_texts) != len(claimed_senders):
            raise ValueError(
                "email_texts and claimed_senders must have the same length."
            )

        # Batch-encode all texts together for efficiency; handle None results.
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
        """Convenience constructor that reads preprocessing config from ExperimentConfig."""
        return cls(
            encoder=encoder,
            head=head,
            store=store,
            preprocessing=config.data.preprocessing,
            device=device,
        )

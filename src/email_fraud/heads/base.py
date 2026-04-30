"""BaseHead ABC.

Heads consume encoder embeddings and make attribution/anomaly decisions.
They bridge the contrastive encoder (which produces metric-space embeddings)
and the scoring pipeline (which needs calibrated confidence scores and tier
labels for operational use).

Architecture overview
---------------------
Training produces a metric space where emails from the same sender cluster
together.  The head exploits this structure at inference time:

  1. fit()   — Called during profile-building (after training, on a reference
               corpus).  For each known sender, aggregates their emails into
               a compact profile (e.g. a centroid).

  2. score() — Called for each query at inference time.  Compares the query's
               embedding against the claimed sender's stored profile and returns
               a calibrated anomaly score.

  3. save()/load() — Profiles are separate from model weights; they need their
               own serialization so they can be updated without retraining.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    """Abstract base for per-sender profiling and anomaly-scoring heads.

    A head is responsible for:
    1. fit()  — building / updating an in-memory sender profile from labeled
                embeddings (called during training and profile-building).
    2. score() — given a query embedding and a claimed sender id, return a
                 calibrated anomaly score, a confidence tier, and an abstain
                 flag (True when the profile has too few samples to be
                 reliable).
    3. save() / load() — serialization so profiles survive restarts.
    """

    @abstractmethod
    def fit(
        self,
        embeddings: torch.Tensor,
        sender_ids: list[str],
    ) -> None:
        """Ingest labeled embeddings and update per-sender profiles in-place.

        Called after each training epoch (or during a dedicated profile-building
        pass over the reference corpus) to accumulate sender representations.

        Args:
            embeddings: (N, d) float tensor, L2-normalized.
            sender_ids: length-N list of sender identifier strings.
                        May contain duplicates (multiple emails per sender).
        """
        ...

    @abstractmethod
    def score(
        self,
        query: torch.Tensor,
        sender_id: str,
    ) -> dict[str, object]:
        """Score a query embedding against a stored sender profile.

        Args:
            query:     (d,) or (1, d) float tensor, L2-normalized.
            sender_id: claimed sender to compare against.

        Returns:
            dict with keys:
            - "score"   : float — higher means *more* like the claimed sender
                          (convention: 1.0 = definitely this sender, 0.0 = unlikely).
            - "tier"    : str   — confidence tier (e.g. "low", "high") based on
                          how many emails the profile was built from.
            - "abstain" : bool  — True when the profile is too sparse to trust
                          (e.g. fewer than 5 emails seen); callers should not
                          make hard fraud/non-fraud decisions in this case.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist profiles to *path*.

        Profiles are stored separately from model weights because they are
        updated frequently (new emails arrive) without retraining the encoder.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore profiles from *path*."""
        ...

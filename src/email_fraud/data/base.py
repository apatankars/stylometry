"""BaseDataset ABC and EpisodeBatch dataclass.

EpisodeBatch is the canonical unit of data flowing through the training loop:
a batch of (embedding, label, sender_id, text) tuples drawn by the PKSampler
to ensure each batch contains exactly P senders × K emails.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeBatch:
    """A P×K structured batch ready for contrastive loss computation.

    Produced by episode_collate() and consumed by the Trainer's training loop.
    The "episode" concept comes from few-shot learning: each batch is treated
    as a small self-contained classification problem where the goal is to group
    emails by sender.

    Attributes:
        embeddings: (P*K, d) float tensor, L2-normalized (populated post-encode).
                    Empty (torch.empty(0)) when the batch first comes out of the
                    DataLoader; the Trainer fills this after calling encoder.encode().
        labels:     (P*K,) integer tensor — sender index within this batch only
                    (0 to P-1, assigned by first-appearance order in episode_collate).
                    These are NOT global sender ids; they're batch-local integers
                    that the loss functions use to identify positives and negatives.
        sender_ids: length P*K list of raw sender id strings (e.g. email addresses).
                    Kept for debugging, logging, and head.fit() calls.
        texts:      length P*K list of raw email text strings.
                    The Trainer tokenizes these via encoder.tokenize().
    """

    embeddings: torch.Tensor
    labels: torch.Tensor
    sender_ids: list[str]
    texts: list[str]


class BaseDataset(Dataset, ABC):
    """Abstract base for email corpora used in contrastive training.

    Subclasses must implement __getitem__ returning (text, sender_id) pairs
    and the sender_ids property so PKSampler can build per-sender index maps.

    The concrete episode_collate classmethod is provided here because the
    collation logic (stacking texts, building integer labels from sender_ids)
    is shared across all dataset implementations.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[str, str]:
        """Return (email_text, sender_id) for *index*."""
        ...

    @property
    @abstractmethod
    def sender_ids(self) -> list[str]:
        """Ordered list of sender ids, one per item in the dataset.

        Used by PKSampler to build per-sender index maps.  The list must be
        the same length as the dataset and in the same order as __getitem__
        (i.e. sender_ids[i] == __getitem__(i)[1]).
        """
        ...

    @classmethod
    def episode_collate(cls, batch: list[tuple[str, str]]) -> EpisodeBatch:
        """Collate a list of (text, sender_id) tuples into an EpisodeBatch.

        Called by DataLoader as the collate_fn.  Converts raw string sender ids
        to integer labels (0..P-1) in order of first appearance within this batch.

        Integer labels are assigned by the order of first appearance of
        sender ids in the batch (deterministic within a single batch).
        Embeddings tensor is empty — the training loop fills it after encoding.

        Why first-appearance order?
        ---------------------------
        The loss functions only care that same-sender items share the same label
        integer; the specific integer value doesn't matter.  First-appearance
        order is deterministic given the batch order (which PKSampler controls),
        so labels are reproducible.
        """
        texts, sender_id_strs = zip(*batch)
        texts = list(texts)
        sender_id_strs = list(sender_id_strs)

        # Map sender_id strings to integers in order of first appearance.
        # e.g. ["alice", "alice", "bob", "alice", "bob"] → [0, 0, 1, 0, 1]
        seen: dict[str, int] = {}
        int_labels: list[int] = []
        for sid in sender_id_strs:
            if sid not in seen:
                seen[sid] = len(seen)
            int_labels.append(seen[sid])

        labels = torch.tensor(int_labels, dtype=torch.long)

        return EpisodeBatch(
            embeddings=torch.empty(0),  # filled by Trainer after encode()
            labels=labels,
            sender_ids=sender_id_strs,
            texts=texts,
        )

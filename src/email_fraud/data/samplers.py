"""PKSampler — produces P×K structured batches for contrastive training.

Each yielded index-list contains exactly P senders × K emails, guaranteeing
every batch has the label structure required by SupConLoss and batch-hard
TripletLoss.

Reference design from: Hermans, Beyer, Leibe "In Defense of the Triplet Loss"
arXiv:1703.07737, Section 2 (batch construction).

Why P×K sampling?
-----------------
Contrastive losses need multiple emails per sender in each batch to form
positive pairs / triplets.  A standard random sampler would frequently produce
batches where a sender appears only once (no positives → no loss signal).

With P=16 senders and K=4 emails each (N=64 per batch), every anchor has
K-1=3 guaranteed positives and (P-1)*K=60 negatives — a rich training signal.

The tradeoff: P and K must satisfy P*K <= batch_size, and each sender must
have >= K emails in the dataset.  Senders with fewer are silently excluded.
"""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


class PKSampler(Sampler[list[int]]):
    """Sample P senders × K emails per batch, without replacement per sender.

    Args:
        sender_ids:  Per-item sender id strings (length N, one per dataset item).
        p:           Number of distinct senders per batch.
        k:           Number of emails per sender per batch.
        drop_last:   If True (default), skip the last partial batch of senders.
        seed:        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        sender_ids: list[str],
        p: int,
        k: int,
        drop_last: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.k = k
        self.drop_last = drop_last
        # Using random.Random (not torch.Generator) so the sampler's state is
        # independent of PyTorch's global RNG and doesn't interfere with model
        # weight initialization or data augmentation.
        self._rng = random.Random(seed)

        # Build per-sender index lists from the flat sender_ids list.
        # defaultdict(list) avoids explicit "if not in" checks.
        self._sender_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sid in enumerate(sender_ids):
            self._sender_to_indices[sid].append(idx)

        # Only keep senders with at least K emails — senders with fewer cannot
        # fill a K-slot without repetition, which would bias the contrastive signal.
        self._eligible_senders: list[str] = [
            sid
            for sid, indices in self._sender_to_indices.items()
            if len(indices) >= k
        ]

        # Hard fail early rather than producing silent empty batches.
        if len(self._eligible_senders) < p:
            raise ValueError(
                f"PKSampler requires at least p={p} senders with >= k={k} emails each; "
                f"only {len(self._eligible_senders)} qualify."
            )

    def __iter__(self):
        # Shuffle senders at the start of each epoch so batch composition varies.
        senders = list(self._eligible_senders)
        self._rng.shuffle(senders)

        # Number of complete P-sized sender groups in this epoch.
        n_batches = len(senders) // self.p
        if not self.drop_last and len(senders) % self.p:
            n_batches += 1

        for batch_idx in range(n_batches):
            batch_senders = senders[batch_idx * self.p : (batch_idx + 1) * self.p]
            if len(batch_senders) < self.p:
                # Last partial group when drop_last=False — skip for now
                # (padding logic not yet implemented).
                continue

            indices: list[int] = []
            for sid in batch_senders:
                # Shuffle this sender's emails and take the first K.
                # Using a copy so the original list isn't mutated between epochs.
                pool = list(self._sender_to_indices[sid])
                self._rng.shuffle(pool)
                # Slicing to K gives "without replacement within a batch" semantics.
                # Across epochs, the same email may appear in different batches.
                indices.extend(pool[: self.k])

            # Yield a flat list of P*K dataset indices.
            # The DataLoader collects these into a batch via episode_collate.
            yield indices

    def __len__(self) -> int:
        # Number of P×K batches yielded per epoch.
        # With a batch_sampler, DataLoader uses this directly as len(dataloader).
        return len(self._eligible_senders) // self.p

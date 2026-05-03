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


# ---------------------------------------------------------------------------
# SyntheticBalancedSampler
# ---------------------------------------------------------------------------

_SYN_SUFFIX = "__syn"


class SyntheticBalancedSampler(Sampler[list[int]]):
    """PKSampler variant that guarantees n_syn synthetic–real sender pairs per batch.

    Each batch of P senders is structured as:
        - n_syn  synthetic senders  (e.g. "alice@enron.com__syn")
        - n_syn  their real counterparts  ("alice@enron.com")
        - P - 2*n_syn  other real senders (no synthetic counterpart in this batch)

    This guarantees that for every batch the contrastive loss directly compares
    real emails against LLM-generated imitations of the same person — the hardest
    possible negatives — rather than relying on random batch composition to produce
    these pairs.

    Senders with a synthetic counterpart are *only* drawn as pairs; they never
    appear solo.  Senders without a synthetic counterpart fill the remaining slots.

    Args:
        sender_ids:    Per-item sender id strings (one per dataset item).
                       Synthetic senders must end with "__syn" and their real
                       counterpart must also be present (e.g. "alice__syn" requires
                       "alice" to exist).
        p:             Total senders per batch (real + synthetic combined).
        k:             Emails per sender per batch.
        n_syn:         Number of synthetic–real pairs per batch.  Must satisfy
                       2 * n_syn <= p.
        drop_last:     Skip the last partial batch if the epoch runs dry.
        seed:          Optional RNG seed.
    """

    def __init__(
        self,
        sender_ids: list[str],
        p: int,
        k: int,
        n_syn: int = 2,
        drop_last: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if 2 * n_syn > p:
            raise ValueError(
                f"2 * n_syn ({2 * n_syn}) must be <= p ({p}); "
                "each synthetic sender occupies one slot and its real counterpart another."
            )
        self.p = p
        self.k = k
        self.n_syn = n_syn
        self.drop_last = drop_last
        self._rng = random.Random(seed)

        # Build per-sender index lists.
        sender_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sid in enumerate(sender_ids):
            sender_to_indices[sid].append(idx)

        def _eligible(sid: str) -> bool:
            return len(sender_to_indices[sid]) >= k

        # Identify synthetic senders and their real counterparts.
        syn_senders = {sid for sid in sender_to_indices if sid.endswith(_SYN_SUFFIX)}
        real_senders = {sid for sid in sender_to_indices if not sid.endswith(_SYN_SUFFIX)}

        # Pairs: (real_sid, syn_sid) where both are eligible.
        self._eligible_pairs: list[tuple[str, str]] = []
        for syn_sid in syn_senders:
            real_sid = syn_sid[: -len(_SYN_SUFFIX)]
            if real_sid in real_senders and _eligible(real_sid) and _eligible(syn_sid):
                self._eligible_pairs.append((real_sid, syn_sid))

        # Real-only senders: real senders that do NOT have a synthetic counterpart.
        # These fill the remaining P - 2*n_syn slots per batch.
        paired_real = {real for real, _ in self._eligible_pairs}
        self._eligible_real_only: list[str] = [
            sid for sid in real_senders if sid not in paired_real and _eligible(sid)
        ]

        slots_real_only = p - 2 * n_syn
        if len(self._eligible_pairs) < n_syn:
            raise ValueError(
                f"SyntheticBalancedSampler requires at least n_syn={n_syn} eligible "
                f"synthetic–real pairs; only {len(self._eligible_pairs)} qualify "
                f"(both sender and its __syn counterpart need >= k={k} emails)."
            )
        if len(self._eligible_real_only) < slots_real_only:
            raise ValueError(
                f"SyntheticBalancedSampler needs {slots_real_only} real-only senders "
                f"(p - 2*n_syn = {p} - {2 * n_syn}) but only "
                f"{len(self._eligible_real_only)} qualify."
            )

        self._sender_to_indices = dict(sender_to_indices)

    def __iter__(self):
        pairs = list(self._eligible_pairs)
        real_only = list(self._eligible_real_only)
        self._rng.shuffle(pairs)
        self._rng.shuffle(real_only)

        slots_real_only = self.p - 2 * self.n_syn
        n_batches = min(len(pairs) // self.n_syn, len(real_only) // slots_real_only)

        for i in range(n_batches):
            batch_pairs = pairs[i * self.n_syn : (i + 1) * self.n_syn]
            batch_real = real_only[i * slots_real_only : (i + 1) * slots_real_only]

            # Interleave: real counterpart immediately before its synthetic twin so
            # episode_collate assigns them adjacent batch-local labels (cosmetic only,
            # loss is label-order invariant).
            batch_senders: list[str] = []
            for real_sid, syn_sid in batch_pairs:
                batch_senders.extend([real_sid, syn_sid])
            batch_senders.extend(batch_real)

            indices: list[int] = []
            for sid in batch_senders:
                pool = list(self._sender_to_indices[sid])
                self._rng.shuffle(pool)
                indices.extend(pool[: self.k])

            yield indices

    def __len__(self) -> int:
        slots_real_only = self.p - 2 * self.n_syn
        return min(
            len(self._eligible_pairs) // self.n_syn,
            len(self._eligible_real_only) // slots_real_only,
        )

"""Pairwise Contrastive Loss (Hadsell et al. 2006).

Reference: Hadsell, Chopra, LeCun "Dimensionality Reduction by Learning an
           Invariant Mapping", CVPR 2006.

For a batch of N embeddings with integer sender labels, all same-sender pairs
are treated as positives (y=1) and all different-sender pairs as negatives
(y=0).  The loss pushes positives together and negatives apart by at least
a margin.

    L = (1/|pairs|) * Σ_{i<j} [ y * d² + (1-y) * relu(m - d)² ]

where d = ||z_i - z_j||_2 (L2 distance on unit-normalized embeddings,
equivalent to sqrt(2 - 2·cos(i,j))).

This is the oldest and simplest of the three contrastive objectives; it
differs from TripletLoss in operating on pairs rather than triplets, and
from SupConLoss in using a hard margin rather than a soft log-sum-exp.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from email_fraud.losses.base import BaseLoss
from email_fraud.registry import register


@register("loss", "contrastive")
class ContrastiveLoss(BaseLoss):
    """Pairwise contrastive loss (Hadsell 2006).

    Args:
        margin:  Minimum distance enforced between negative pairs (default 1.0).
                 With L2-normed vectors ||z||=1, the max distance is 2, so
                 margin ∈ (0, 2].
        mining:  'all' = all pairs; 'semi_hard' = only semi-hard negatives
                 (negatives closer than margin but farther than hardest positive).
    """

    def __init__(self, margin: float = 1.0, mining: str = "all") -> None:
        super().__init__()
        if mining not in {"all", "semi_hard"}:
            raise ValueError(f"Unknown mining strategy '{mining}'. Choose 'all' or 'semi_hard'.")
        self.margin = margin
        self.mining = mining

    @property
    def requires_pk_sampler(self) -> bool:
        # PKSampler is required to guarantee positives in every batch.
        # With random batching, many batches would have only one email per
        # sender (K=1), leaving no positive pairs and no training signal.
        return True

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise contrastive loss over a PK-structured batch.

        Args:
            embeddings: (N, d) L2-normalized float tensor.
            labels:     (N,) integer sender ids.

        Returns:
            Scalar mean loss over all valid pairs.
        """
        n = embeddings.size(0)
        device = embeddings.device

        # Pairwise L2 distances using the unit-sphere identity:
        #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a·b) = 2 - 2*(a·b)
        # This is faster than computing differences explicitly and numerically
        # equivalent for L2-normalized vectors.
        dot = torch.mm(embeddings, embeddings.T)          # (N, N)
        dist_sq = (2.0 - 2.0 * dot).clamp(min=0.0)       # (N, N)  clamp avoids sqrt(negative) from float rounding
        dist = dist_sq.sqrt()                              # (N, N)  actual L2 distances

        # Build positive / negative masks (upper triangle only to avoid double-counting)
        # same[i,j] = True iff labels[i] == labels[j]
        labels_col = labels.unsqueeze(1)                   # (N, 1)
        labels_row = labels.unsqueeze(0)                   # (1, N)
        same = (labels_col == labels_row)                  # (N, N)
        # Exclude self-pairs (distance = 0, always positive but uninformative)
        eye = torch.eye(n, dtype=torch.bool, device=device)
        same = same & ~eye
        diff = ~same & ~eye

        # triu(diagonal=1) selects the strict upper triangle so each pair
        # is counted exactly once (avoids double-counting (i,j) and (j,i)).
        triu = torch.ones(n, n, dtype=torch.bool, device=device).triu(diagonal=1)
        pos_mask = same & triu
        neg_mask = diff & triu

        if self.mining == "semi_hard":
            # Semi-hard mining strategy (Schroff et al. FaceNet 2015):
            # Only include negatives that are:
            #   (a) outside the hardest positive's distance (farther than the
            #       farthest same-class sample) — so the negative isn't trivially easy,
            #   (b) inside the margin — so the negative still produces a non-zero loss.
            # This avoids "easy negatives" (already well-separated) and "hard negatives"
            # (collapsed cases) that can destabilize training.
            pos_dist_full = dist.masked_fill(~same, 0.0).max(dim=1).values  # (N,)  hardest positive per anchor
            hard_pos_ref = pos_dist_full.unsqueeze(1).expand(n, n)           # (N, N)
            semi_hard = (dist > hard_pos_ref) & (dist < self.margin)
            neg_mask = neg_mask & semi_hard

        pos_dist = dist[pos_mask]
        neg_dist = dist[neg_mask]

        # Guard against empty batches (e.g. all singletons after mining)
        if pos_dist.numel() == 0 and neg_dist.numel() == 0:
            # Return zero with a grad_fn so backward() doesn't crash.
            return embeddings.sum() * 0.0

        # Positive loss: pull same-sender pairs together (minimize d²)
        pos_loss = pos_dist.pow(2) if pos_dist.numel() > 0 else torch.zeros(1, device=device)
        # Negative loss: push different-sender pairs apart by at least margin.
        # relu(m - d) is zero when d >= margin (already separated), non-zero otherwise.
        neg_loss = F.relu(self.margin - neg_dist).pow(2) if neg_dist.numel() > 0 else torch.zeros(1, device=device)

        # Average both terms equally so the loss scale doesn't depend on
        # the positive/negative imbalance within a batch.
        return (pos_loss.mean() + neg_loss.mean()) / 2.0

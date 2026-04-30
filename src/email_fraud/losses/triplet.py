"""Triplet Loss with batch-hard mining.

Reference: Hermans, Beyer, Leibe "In Defense of the Triplet Loss for
           Person Re-Identification", arXiv:1703.07737.

Batch-hard mining selects, for each anchor i:
  - hardest positive: the same-class sample with *maximum* distance to i.
  - hardest negative: the different-class sample with *minimum* distance to i.

Why batch-hard over random triplet mining?
------------------------------------------
Random mining produces mostly "easy" triplets — pairs that are already well
separated — giving near-zero loss and slow learning.  Batch-hard forces the
model to focus on the most informative (hardest) pairs in each batch.

Why squared Euclidean distance?
--------------------------------
For L2-normalized vectors, squared Euclidean distance is a monotone transform
of cosine distance (d² = 2 - 2·cos), so the ranking is identical.  Using d²
avoids the sqrt, which has infinite gradient at 0 and can cause NaNs during
early training when embeddings collapse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from email_fraud.losses.base import BaseLoss

# @register("loss", "triplet") — Hermans et al. arXiv:1703.07737
from email_fraud.registry import register


@register("loss", "triplet")
class TripletLoss(BaseLoss):
    """Triplet loss with batch-hard or all-pairs mining.

    Operates on L2-normalized embeddings; uses squared Euclidean distance
    (equivalent to 2 - 2*cosine_similarity for unit vectors).
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining: str = "batch_hard",
    ) -> None:
        super().__init__()
        if mining not in {"batch_hard", "all"}:
            raise ValueError(
                f"Unknown mining strategy '{mining}'. Choose 'batch_hard' or 'all'."
            )
        self.margin = margin
        self.mining = mining

    @property
    def requires_pk_sampler(self) -> bool:
        # PKSampler ensures K >= 2 same-sender emails per anchor, so there is
        # always a valid positive to form a triplet.
        return True

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss over a PK-structured batch.

        Args:
            embeddings: (N, d) L2-normalized float tensor.
            labels:     (N,) integer sender ids.

        Returns:
            Scalar mean hinge loss over valid (non-zero) triplets.
        """
        # Pairwise squared Euclidean distance matrix (N, N).
        # For L2-normed vectors: ||a - b||^2 = 2 - 2*(a·b)
        # Clamped to avoid sqrt(negative) from floating-point rounding errors.
        dot = torch.mm(embeddings, embeddings.T)          # (N, N)
        dist = (2.0 - 2.0 * dot).clamp(min=0.0)          # (N, N) squared distances

        if self.mining == "batch_hard":
            return self._batch_hard(dist, labels)
        return self._all_pairs(dist, labels)

    # ------------------------------------------------------------------
    # Mining strategies
    # ------------------------------------------------------------------

    def _batch_hard(
        self,
        dist: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-hard: hardest positive + hardest negative per anchor.

        For each anchor i:
          - hardest positive: argmax_{j: y_j==y_i, j≠i} d(i,j)
          - hardest negative: argmin_{k: y_k≠y_i} d(i,k)

        The loss for anchor i is: relu(d_pos - d_neg + margin)
        A triplet "violates" the margin when d_pos - d_neg + margin > 0,
        i.e. the positive is farther than the negative by less than the margin.
        """
        n = dist.size(0)
        device = dist.device

        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        # pos_mask[i,j] = 1 iff same sender (includes diagonal at this point)
        pos_mask = (labels_col == labels_row).float()  # (N, N)
        neg_mask = 1.0 - pos_mask                      # (N, N)

        # Remove self from positive distances (diagonal = 0, would always be "hardest")
        eye = torch.eye(n, device=device)
        pos_mask = pos_mask - eye

        # Hardest positive: mask non-positives with 0, take max per row.
        # Note: using * instead of masked_fill so zeros don't contaminate the max.
        hardest_pos = (dist * pos_mask).max(dim=1).values  # (N,)

        # Hardest negative: mask non-negatives with a value larger than any distance,
        # then take min per row.  This ensures only true negatives compete.
        large = dist.max().item() + 1.0
        hardest_neg = (dist + large * (1.0 - neg_mask)).min(dim=1).values  # (N,)

        # Hinge loss: penalize only violated triplets (d_pos - d_neg + margin > 0)
        losses = F.relu(hardest_pos - hardest_neg + self.margin)

        # Only average over anchors with at least one valid positive.
        # An anchor without a positive (shouldn't happen with PKSampler) would
        # have hardest_pos = 0 and could bias the loss.
        valid = pos_mask.sum(dim=1) > 0
        if not valid.any():
            return losses.mean() * 0.0
        return losses[valid].mean()

    def _all_pairs(
        self,
        dist: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """All-pairs triplet loss — exhaustive (i, j, k) triplets.

        Unlike batch-hard, this evaluates every valid (anchor, positive, negative)
        combination in the batch.  This is O(N³) in memory, so only suitable
        for small N (typically N = P*K <= 128).

        Useful for small batches or debugging where you want the full loss
        landscape, not just the batch-hard approximation.
        """
        n = dist.size(0)
        device = dist.device
        eye = torch.eye(n, device=device)

        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        # pos_mask[i,j] = 1 iff same sender AND i != j
        pos_mask = (labels_col == labels_row).float() - eye
        # neg_mask[i,k] = 1 iff different sender
        neg_mask = (labels_col != labels_row).float()

        # Broadcast to build all (N, N, N) triplets efficiently:
        # triplet_loss[i, j, k] = relu(d(i,j) - d(i,k) + margin)
        # where j is a positive and k is a negative for anchor i.
        pos_dist = dist.unsqueeze(2)   # (N, N, 1) — broadcast over negatives
        neg_dist = dist.unsqueeze(1)   # (N, 1, N) — broadcast over positives
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)  # (N, N, N)

        # valid_mask[i,j,k] = 1 iff j is a valid positive for i AND k is a valid negative for i
        valid_mask = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)  # (N, N, N)
        # Average only over valid triplets to avoid dividing by zero
        losses = (triplet_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return losses

"""Supervised Contrastive Loss (L_sup_out variant).

Reference: Khosla et al. "Supervised Contrastive Learning"
           NeurIPS 2020, arXiv:2004.11362.

The L_sup_out variant places the 1/|P(i)| normalisation factor *outside* the
log (equation 4 in the paper), which is more numerically stable in practice
than the L_sup_in variant.

How it works
------------
For each anchor i, we want to maximize the similarity to all other in-batch
samples that share i's label (its "positives" P(i)) relative to all other
samples A(i).  Formally:

    L_i = -(1/|P(i)|) * Σ_{p ∈ P(i)} log [ exp(z_i · z_p / τ) / Σ_{a ∈ A(i)} exp(z_i · z_a / τ) ]

where τ is the temperature, z are L2-normalized embeddings (so z_i · z_j = cos(z_i, z_j)).

The log of the denominator is computed with a log-sum-exp for numerical stability
(prevents overflow when similarities are large).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from email_fraud.losses.base import BaseLoss

# @register("loss", "supcon") — Khosla et al. NeurIPS 2020 (arXiv:2004.11362)
from email_fraud.registry import register


@register("loss", "supcon")
class SupConLoss(BaseLoss):
    """Supervised contrastive loss (L_sup_out).

    For each anchor i, the loss averages -log P(z_p | z_i) over all positives
    p ∈ P(i), where P(i) is the set of in-batch samples sharing i's label.
    The denominator sums over all other samples A(i) = {1..N} \\ {i}.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        # Lower temperature → sharper softmax → harder negatives matter more.
        # 0.07 is the SimCLR default; 0.1 is a safe starting point for email data.
        self.temperature = temperature

    @property
    def requires_pk_sampler(self) -> bool:
        # With random batching, many anchors would have no in-batch positives
        # (K=1 for their sender), making the loss undefined or trivially zero.
        # PKSampler guarantees K >= 2 emails per sender in every batch.
        return True

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L_sup_out over a PK-structured batch.

        Args:
            embeddings: (N, d) L2-normalized float tensor.
            labels:     (N,) integer sender ids.

        Returns:
            Scalar mean loss.
        """
        device = embeddings.device
        n = embeddings.size(0)

        # Cosine similarity matrix scaled by 1/τ.
        # Since embeddings are L2-normalized, dot product == cosine similarity.
        # Scaling before exp() is numerically equivalent to scaling inside,
        # but doing it here lets us reuse sim for both numerator and denominator.
        sim = torch.mm(embeddings, embeddings.T) / self.temperature  # (N, N)

        # pos_mask[i,j] = 1 iff labels[i] == labels[j] AND i != j
        labels_col = labels.unsqueeze(1)       # (N, 1)
        labels_row = labels.unsqueeze(0)       # (1, N)
        pos_mask = (labels_col == labels_row).float()  # (N, N)  includes self
        diag_mask = torch.eye(n, device=device)
        pos_mask = pos_mask - diag_mask  # zero out the diagonal (self is not a positive)

        # num_pos[i] = number of in-batch positives for anchor i
        num_pos = pos_mask.sum(dim=1)  # (N,)
        # Skip anchors that have no positives in this batch (shouldn't happen
        # with PKSampler, but defensive guard).
        valid = num_pos > 0

        if not valid.any():
            # Return zero with a grad_fn attached so backward() doesn't crash.
            return embeddings.sum() * 0.0

        # Log-sum-exp denominator: Σ_{a ∈ A(i)} exp(z_i · z_a / τ)
        # A(i) = all samples except i itself → zero out diagonal with self_mask.
        self_mask = 1.0 - diag_mask   # (N, N) — ones everywhere except diagonal
        log_denom = torch.log(
            # sum over j≠i of exp(sim[i,j]); clamp to avoid log(0)
            (self_mask * torch.exp(sim)).sum(dim=1, keepdim=True).clamp(min=1e-9)
        )  # (N, 1)

        # log P(z_j | z_i) = sim[i,j] - log_denom[i]  (broadcast over j)
        log_prob = sim - log_denom    # (N, N)

        # Mean log-probability over positives for each anchor:
        # (1/|P(i)|) * Σ_{p ∈ P(i)} log_prob[i, p]
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / num_pos.clamp(min=1)

        # Final loss is the negative mean (we want to *maximize* log-probability
        # of positives, so the loss to *minimize* is the negative of that).
        # Only average over valid anchors (those with ≥ 1 positive).
        loss = -mean_log_prob_pos[valid].mean()
        return loss

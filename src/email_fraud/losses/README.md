# losses/

Contrastive loss functions that train the encoder to pull same-sender emails together and push different-sender emails apart in the embedding space.

---

## How contrastive training works

For a batch of N emails (each a `(d,)` L2-normalized embedding), the loss function:
1. Uses sender labels to identify **positive pairs** (same sender) and **negative pairs** (different senders)
2. Computes a scalar loss that is small when positives are close and negatives are far apart
3. Gradients flow back through the encoder, adjusting weights to improve clustering

All losses operate on the unit hypersphere (L2-normalized embeddings), so distances are bounded: cosine similarity ∈ [-1, 1], L2 distance ∈ [0, 2].

---

## Files

### `base.py` — `BaseLoss`

Abstract base class. All losses must:
- Implement `forward(embeddings, labels) → scalar Tensor`
- Declare `requires_pk_sampler` (True for all three current losses)

`requires_pk_sampler = True` tells the training script to use `PKSampler` so every batch contains `P` senders × `K` emails each — guaranteeing at least one positive pair per anchor.

---

### `supcon.py` — `SupConLoss`

**Supervised Contrastive Loss** (Khosla et al., NeurIPS 2020, arXiv:2004.11362).

Registered as `"supcon"`. **Recommended default.**

#### Formula (L_sup_out variant)

For each anchor `i`:

```
L_i = -(1/|P(i)|) * Σ_{p ∈ P(i)} log [ exp(z_i · z_p / τ) / Σ_{a ≠ i} exp(z_i · z_a / τ) ]
```

where `P(i)` = in-batch emails sharing `i`'s sender label, `τ` = temperature.

#### Intuition
- Maximizes the probability of drawing a positive when sampling uniformly from the rest of the batch
- Temperature `τ` controls the sharpness: lower → harder negatives dominate more

#### Key parameter
| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 0.1 | Lower = sharper contrastive distribution |

---

### `triplet.py` — `TripletLoss`

**Triplet loss with batch-hard mining** (Hermans et al., arXiv:1703.07737).

Registered as `"triplet"`.

#### Formula

For each anchor `i`:
```
L_i = relu(d(i, hardest_pos) - d(i, hardest_neg) + margin)
```

where `d` is squared Euclidean distance (= `2 - 2·cos` for unit vectors).

#### Mining strategies

| Strategy | Description |
|----------|-------------|
| `"batch_hard"` | Hardest positive (max distance) + hardest negative (min distance) per anchor |
| `"all"` | All valid (anchor, positive, negative) triplets — O(N³), use for small batches |

#### Key parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `margin` | 0.3 | Minimum required gap between positive and negative distances |
| `mining` | `"batch_hard"` | Triplet selection strategy |

---

### `contrastive.py` — `ContrastiveLoss`

**Pairwise contrastive loss** (Hadsell et al., CVPR 2006). The original contrastive loss.

Registered as `"contrastive"`.

#### Formula

```
L = (1/|pairs|) * Σ_{i<j} [ y·d² + (1-y)·relu(m - d)² ]
```

where `y=1` for same-sender pairs, `y=0` for different-sender pairs, `d` = L2 distance.

#### Mining strategies

| Strategy | Description |
|----------|-------------|
| `"all"` | All positive and negative pairs in the batch |
| `"semi_hard"` | Negatives farther than the hardest positive but inside the margin |

Semi-hard mining (Schroff et al. FaceNet 2015) avoids trivially easy negatives that are already well-separated.

#### Key parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `margin` | 1.0 | Distance at which negatives stop contributing to loss |
| `mining` | `"all"` | Pair selection strategy |

---

## Comparison

| Loss | Objective | Complexity | Best for |
|------|-----------|------------|----------|
| SupCon | Maximize similarity to all positives jointly | O(N²) | Large K, rich positive signal |
| Triplet (batch-hard) | Separate hardest positive from hardest negative | O(N²) | When convergence is slow with SupCon |
| Contrastive | Push all pairs together / apart by margin | O(N²) | Simpler baseline; pairs are cheaper than triplets |

---

## Configuration reference

```yaml
loss:
  name: supcon        # supcon | triplet | contrastive
  temperature: 0.1    # SupConLoss only
  margin: 0.3         # TripletLoss / ContrastiveLoss
  mining: batch_hard  # batch_hard | all (triplet) / all | semi_hard (contrastive)
```

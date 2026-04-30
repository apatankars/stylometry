# email_fraud — source package

Contrastive learning system for email sender verification and fraud detection.

---

## What this does

Given an email claiming to be from a known sender, the system returns a score in `[0, 1]` indicating how consistent the email is with that sender's historical writing style.

- **Score near 1.0** → email is stylistically consistent with the claimed sender
- **Score near 0.0** → email is a significant outlier — possible impersonation / fraud
- **`abstain=True`** → not enough historical emails to make a reliable judgment

---

## Architecture overview

```
┌────────────────────────────────────────────────────────────────┐
│                         TRAINING                               │
│                                                                │
│  EnronDataset ──► PKSampler ──► EpisodeBatch                  │
│       (P senders × K emails per batch)                         │
│                          │                                     │
│                    HFEncoder.encode()                          │
│                          │  (B, d) L2-normalized embeddings    │
│                          ▼                                     │
│              SupConLoss / TripletLoss / ContrastiveLoss        │
│                          │  scalar loss                        │
│                          ▼                                     │
│                    AdamW + scheduler                           │
│                    + AMP + grad clip                           │
│                     (Trainer)                                  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                      PROFILE BUILDING                          │
│                                                                │
│  Reference corpus ──► HFEncoder.encode() ──► head.fit()       │
│                                               (centroid + spread per sender)
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                        INFERENCE                               │
│                                                                │
│  raw email + claimed sender                                    │
│       │                                                        │
│       ▼                                                        │
│  preprocessing (strip replies/sigs, entity mask)              │
│       │                                                        │
│       ▼                                                        │
│  HFEncoder.encode() → (d,) embedding                          │
│       │                                                        │
│       ▼                                                        │
│  PrototypicalHead.score(query, sender_id)                     │
│       │  cosine distance → z-score → [0, 1] score             │
│       ▼                                                        │
│  ScoringResult { score, tier, abstain, embedding }            │
└────────────────────────────────────────────────────────────────┘
```

---

## Package structure

| Directory | Purpose |
|-----------|---------|
| [`encoders/`](encoders/README.md) | Text → embedding (HuggingFace transformer + LoRA + pooling) |
| [`losses/`](losses/README.md) | Contrastive training objectives (SupCon, Triplet, Contrastive) |
| [`heads/`](heads/README.md) | Per-sender profiling and anomaly scoring (prototypical centroid) |
| [`training/`](training/README.md) | Full training loop with checkpointing, AMP, wandb |
| [`data/`](data/README.md) | Dataset loading, email preprocessing, P×K batch sampling |
| [`profiles/`](profiles/README.md) | EWMA centroid store for inference-time profile updates |
| [`scoring/`](scoring/README.md) | End-to-end inference pipeline + PAN evaluation metrics |
| [`utils/`](utils/README.md) | Logging and RunPod GPU management |
| `config.py` | Pydantic v2 config schema + YAML loader |
| `registry.py` | Component registry (encoder/loss/head/dataset lookup by name) |

---

## Top-level modules

### `config.py`

Single source of truth for all experiment settings. Configs are composed by deep-merging `configs/base.yaml` (defaults) with an experiment YAML (overrides):

```python
config = load_config("configs/experiments/roberta_supcon.yaml")
config.encoder.model_name_or_path  # → "roberta-base"
config.training.lr                 # → 2e-5
```

All sub-configs (`EncoderConfig`, `LossConfig`, etc.) use Pydantic v2 with `extra="forbid"` — typos in YAML keys raise a `ValidationError` at load time before any GPU memory is allocated.

### `registry.py`

Turns YAML name strings into concrete Python classes without any if/elif chains in the training script:

```python
EncoderCls = resolve("encoder", config.encoder.name)   # → HFEncoder
LossCls    = resolve("loss",    config.loss.name)       # → SupConLoss
HeadCls    = resolve("head",    config.head.name)       # → PrototypicalHead
```

Components register themselves at class-definition time:

```python
@register("encoder", "hf")
class HFEncoder(BaseEncoder): ...
```

---

## Quick start

```python
from email_fraud.config import load_config
from email_fraud.registry import resolve
from email_fraud.scoring.pipeline import ScoringPipeline
from email_fraud.profiles.store import SenderProfileStore

config   = load_config("configs/experiments/roberta_supcon.yaml")
Encoder  = resolve("encoder", config.encoder.name)
Head     = resolve("head",    config.head.name)

encoder  = Encoder(config.encoder)
head     = Head(confidence_tiers=config.confidence_tiers)
store    = SenderProfileStore(confidence_tiers=config.confidence_tiers)

# Load trained weights
import torch
ckpt = torch.load("runs/roberta_supcon/checkpoint_best.pt", map_location="cpu")
encoder.load_state_dict(ckpt["model_state_dict"])

# Load sender profiles
head.load("profiles/enron_profiles.pkl")

pipeline = ScoringPipeline(encoder, head, store, config.data.preprocessing)
result   = pipeline.score(email_body, claimed_sender="john.doe@company.com")
print(result.score, result.tier, result.abstain)
```

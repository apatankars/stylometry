# Email Fraud Detection — Sender Fingerprinting via Contrastive Learning

Trains a contrastive encoder to build per-sender behavioral profiles and flags
emails that deviate from a sender's established style.  Everything is config-
driven via YAML; swapping encoders, losses, and heads requires no source changes.

---

## Architecture overview

```
Raw email text
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  strip quotes · strip signatures · entity masking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Encoder      │  HFEncoder (RoBERTa / LUAR / ModernBERT / CANINE …)
│  (BaseEncoder)  │  → (B, d) L2-normalised embeddings
└────────┬────────┘
         │
    ┌────┴────────────────────────┐
    │ Training              Inference
    ▼                            ▼
┌────────┐              ┌─────────────────┐
│  Loss  │              │  ScoringPipeline│
│ SupCon │              │  head.score()   │
│ Triplet│              └────────┬────────┘
└────────┘                       │
    │                            ▼
    ▼                  ┌──────────────────┐
┌────────┐             │ SenderProfileStore│
│  Head  │ ──fit()──►  │  (EWMA centroid) │
│ Prototypical│        └──────────────────┘
└────────┘
```

---

## Component guide

### 1. Registry (`src/email_fraud/registry.py`)

The registry is the backbone of the config-driven design.  Every component
class self-registers with a decorator, then the training script resolves
names from YAML to actual classes.

```python
from email_fraud.registry import register, resolve, list_components

# Register a new encoder (done once in the class definition file)
@register("encoder", "my_encoder")
class MyEncoder(BaseEncoder):
    ...

# Resolve at runtime from a config string
EncoderClass = resolve("encoder", "my_encoder")
enc = EncoderClass(cfg.encoder)

# Inspect what's available
print(list_components())
# {'encoder': ['hf', 'my_encoder'], 'loss': ['supcon', 'triplet'], ...}
```

Kinds: `"encoder"`, `"loss"`, `"head"`, `"dataset"`.

---

### 2. Config (`src/email_fraud/config.py`)

Pydantic v2 models with `extra="forbid"` (typos in YAML are caught at load
time, not silently ignored).

```python
from email_fraud.config import load_config

cfg = load_config("configs/experiments/example_luar.yaml")
# configs/base.yaml is merged first; experiment keys win.

cfg.encoder.model_name_or_path   # "rrivera1849/LUAR-MUD"
cfg.loss.temperature             # 0.1
cfg.training.epochs              # 20
```

**Adding a new config field:** add it to the relevant Pydantic sub-model in
`config.py`, set a default, add the field to `configs/base.yaml`.

---

### 3. Encoders (`src/email_fraud/encoders/`)

| Class | Key | Description |
|-------|-----|-------------|
| `HFEncoder` | `"hf"` | Loads any `AutoModel` checkpoint; supports LoRA and three pooling modes |

**Pooling modes** (set `encoder.pooling` in YAML):
- `"mean"` — attention-masked mean over tokens (default).
- `"cls"` — first `[CLS]` token.
- `"luar_episode"` — episode-level mean-pool for LUAR: input shape `(B, K, L)` → `(B, d)`.

**Adding a new encoder:**

```python
# src/email_fraud/encoders/my_encoder.py
from email_fraud.encoders.base import BaseEncoder
from email_fraud.registry import register

@register("encoder", "char_cnn")  # Paper: Zhang et al. 2015
class CharCNNEncoder(BaseEncoder):
    MODEL_TYPE = "char"

    def __init__(self, config):
        super().__init__()
        # ...

    @property
    def embedding_dim(self): return 256

    def encode(self, input_ids, attention_mask, **kwargs):
        # ... return F.normalize(out, p=2, dim=-1)

    def tokenize(self, texts):
        # ... return {"input_ids": ..., "attention_mask": ...}
```

Then import it in `encoders/__init__.py` to fire the `@register` side-effect.

---

### 4. Losses (`src/email_fraud/losses/`)

| Class | Key | Paper |
|-------|-----|-------|
| `SupConLoss` | `"supcon"` | Khosla et al. NeurIPS 2020 (arXiv:2004.11362) |
| `TripletLoss` | `"triplet"` | Hermans et al. arXiv:1703.07737 |

Both require a PK-structured batch (PKSampler).

```python
from email_fraud.losses.supcon import SupConLoss
import torch, torch.nn.functional as F

emb    = F.normalize(torch.randn(16, 768), dim=-1)  # 4 senders × 4 emails
labels = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])

loss = SupConLoss(temperature=0.1)(emb, labels)
```

---

### 5. Heads (`src/email_fraud/heads/`)

| Class | Key | Description |
|-------|-----|-------------|
| `PrototypicalHead` | `"prototypical"` | Centroid + cosine z-score; online fit() |
| `CrossEncoderHead` | `"cross_encoder"` | Reranker stub (not yet implemented) |

```python
from email_fraud.heads.prototypical import PrototypicalHead

head = PrototypicalHead(confidence_tiers=cfg.confidence_tiers)

# Build profile from training embeddings
head.fit(embeddings, sender_ids=["alice", "alice", "bob", "bob"])

# Score a new query
result = head.score(query_embedding, "alice")
# {"score": 0.87, "tier": "high", "abstain": False}

head.save("checkpoints/run_xyz/head.pkl")
head.load("checkpoints/run_xyz/head.pkl")
```

The `score` dict fields:
- `score` — float in [0, 1]; higher = more consistent with the sender's style.
- `tier` — confidence tier based on how many emails are in the profile.
- `abstain` — `True` when `tier == "low"` (< 5 emails; don't trust the score).

---

### 6. Data (`src/email_fraud/data/`)

#### PKSampler

Produces PK-structured batches for contrastive losses.

```python
from email_fraud.data.samplers import PKSampler
from torch.utils.data import DataLoader

sampler = PKSampler(
    sender_ids=dataset.sender_ids,  # one string per item in dataset
    p=4,    # senders per batch
    k=16,   # emails per sender
)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.episode_collate)
```

#### Preprocessing

```python
from email_fraud.data.preprocessing import preprocess
from email_fraud.config import PreprocessingConfig

cfg = PreprocessingConfig(strip_quoted=True, strip_signatures=True, entity_masking=False)
clean_text = preprocess(raw_email_body, cfg)
```

#### Implementing a custom dataset

```python
from email_fraud.data.base import BaseDataset
from email_fraud.registry import register

@register("dataset", "enron")
class EnronDataset(BaseDataset):
    def __init__(self, config):
        self._items = [...]  # list of (text, sender_id)

    def __getitem__(self, idx):
        return self._items[idx]   # (text, sender_id)

    def __len__(self):
        return len(self._items)

    @property
    def sender_ids(self):
        return [s for _, s in self._items]
```

---

### 7. Sender Profile Store (`src/email_fraud/profiles/store.py`)

Lightweight in-memory store with EWMA centroid updates.

```python
from email_fraud.profiles.store import SenderProfileStore
import numpy as np

store = SenderProfileStore(ewma_alpha=0.1, confidence_tiers=cfg.confidence_tiers)

store.upsert("alice@corp.com", np.random.randn(768).astype(np.float32))
store.upsert("alice@corp.com", np.random.randn(768).astype(np.float32))

profile = store.get_profile("alice@corp.com")
# {"centroid": ndarray, "spread": float, "k": int, "metadata": {}}

tier = store.confidence_tier("alice@corp.com")   # "low" / "medium" / "high" / "very_high"

store.save("profiles.json")
store.load("profiles.json")
```

---

### 8. Scoring Pipeline (`src/email_fraud/scoring/pipeline.py`)

End-to-end inference from raw text to anomaly score.

```python
from email_fraud.scoring.pipeline import ScoringPipeline

pipeline = ScoringPipeline(
    encoder=encoder,   # trained HFEncoder
    head=head,         # fitted PrototypicalHead
    store=store,       # SenderProfileStore with profiles loaded
    preprocessing=cfg.data.preprocessing,
    device="cuda",
)

result = pipeline.score(
    email_text="Hi, please approve the wire transfer...",
    claimed_sender="cfo@company.com",
)
print(result.score)    # 0.23  ← low similarity to CFO's known style
print(result.tier)     # "high"
print(result.abstain)  # False

# Batch scoring
results = pipeline.score_batch(texts, senders)
```

---

### 9. Trainer (`src/email_fraud/training/trainer.py`)

```python
from email_fraud.training.trainer import Trainer

trainer = Trainer(
    model=encoder,
    loss_fn=SupConLoss(temperature=0.1),
    head=head,
    config=cfg.training,
    wandb_config=cfg.wandb,
    device="cuda",
)
trainer.train(train_loader, val_loader)
# Logs train/loss and val/loss to wandb each epoch.
# Saves checkpoints/run_<wandb_id>/epoch_NNN.pt
```

---

## Running an experiment

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set credentials
cp .env.example .env
# Edit .env with your WANDB_API_KEY and HF_TOKEN

# 3. Download data (see data/README.md)

# 4. Implement EnronDataset (src/email_fraud/data/enron.py)
#    following the BaseDataset ABC

# 5. Train
python scripts/train.py --config configs/experiments/example_luar.yaml

# 6. Evaluate
python scripts/evaluate.py \
    --config configs/experiments/example_luar.yaml \
    --checkpoint checkpoints/<run_id>/epoch_020.pt \
    --data-dir data/processed/pan

# Add --wandb to also report the PAN metrics to the configured W&B project.
```

---

## Adding a new experiment

1. Copy `configs/experiments/example_luar.yaml` to `configs/experiments/my_exp.yaml`.
2. Change only the fields that differ from `base.yaml`.
3. If you need a new component, create the class with `@register(kind, name)` and import it in the subpackage `__init__.py`.
4. Run with `--config configs/experiments/my_exp.yaml`.

---

## Confidence tiers

The `confidence_tiers` mapping in the config translates the number of training
emails seen for a sender (`k`) into a human-readable confidence tier:

| k range | Tier | Guidance |
|---------|------|---------|
| 1–4 | `low` | Abstain — too few samples |
| 5–9 | `medium` | Score with caution |
| 10–24 | `high` | Reliable |
| 25+ | `very_high` | High confidence |

The head sets `abstain=True` when the tier is `"low"`.

---

## Stub / TODO inventory

| Location | What's missing |
|----------|----------------|
| `encoders/hf_encoder.py` encode() | Full forward pass (currently calls backbone but the luar_episode path needs testing with a real LUAR checkpoint) |
| `heads/prototypical.py` mahalanobis_score() | Ledoit-Wolf covariance estimation |
| `heads/cross_encoder.py` | Full cross-encoder reranker |
| `training/trainer.py` _validate() | Validation loss only; PAN verification metrics live in `scripts/evaluate.py` |
| `scripts/train.py` | Dataset loading (implement BaseDataset subclass first) |
| `scripts/evaluate.py` | Eval pair loading from disk |
| `profiles/store.py` mahalanobis_score() | Full Mahalanobis with pgvector backend |

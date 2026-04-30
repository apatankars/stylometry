# encoders/

Text encoders convert raw email text into fixed-size embedding vectors in a learned metric space, where emails from the same sender cluster together.

---

## Role in the pipeline

```
raw email text
     │
     ▼
 tokenize()        ← encoder.tokenize(texts)
     │
     ▼
 backbone (transformer)
     │
     ▼
 pooling (mean / cls / luar_episode)
     │
     ▼
 projection head   ← optional linear layer
     │
     ▼
 L2 normalize
     │
     ▼
(B, d) embedding   ← passed to loss function or head
```

The encoder is the only component that is **trained** via the contrastive loss. After training, its weights are frozen and used to embed emails for profiling and scoring.

---

## Files

### `base.py` — `BaseEncoder`

Abstract base class that all encoders must inherit from.

| Member | Purpose |
|--------|---------|
| `MODEL_TYPE` | Class-level string tag (e.g. `"subword"`) for architecture branching without `isinstance` checks |
| `encode(input_ids, attention_mask)` | Abstract — returns `(B, d)` **L2-normalized** float tensor |
| `embedding_dim` | Abstract property — output dimensionality |
| `tokenize(texts)` | Abstract — returns HuggingFace-style `dict[str, Tensor]` |

The contract that all embeddings are L2-normalized is enforced by the interface (not checked at runtime). This means cosine similarity = dot product everywhere downstream.

---

### `hf_encoder.py` — `HFEncoder`

The only concrete encoder. Wraps any HuggingFace `AutoModel` (RoBERTa, LUAR, ModernBERT, CANINE, etc.) and registered under the name `"hf"`.

#### Construction

```python
HFEncoder(config: EncoderConfig)
```

On init:
1. Loads `AutoTokenizer` and `AutoModel` from `config.model_name_or_path`
2. Optionally wraps backbone with **LoRA** adapters (`config.lora`)
3. Optionally **freezes** the backbone (`config.freeze_backbone`)
4. Optionally adds a **projection head** (`config.projection_dim`)

#### Pooling strategies

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `"mean"` | Attention-masked mean over token dimension | Default; robust for variable-length emails |
| `"cls"` | First `[CLS]` token | When backbone was pre-trained with NSP/sentence objectives |
| `"luar_episode"` | Mean over K emails then over tokens | LUAR model; input must be `(B, K, L)` |

#### Key methods

```python
encode(input_ids, attention_mask) → (B, d)  # L2-normalized
tokenize(texts: list[str])        → dict[str, Tensor]
```

#### LoRA fine-tuning

When `config.lora` is set, only the low-rank adapter weights are trainable (~0.3% of parameters for RoBERTa-base with `r=8`). This is the recommended training mode — full fine-tuning risks overfitting on the small Enron corpus.

#### LUAR episode pooling

LUAR (Rivera et al. 2021, arXiv:2107.10882) encodes a group of K emails from the same author jointly:

```
(B, K, L) → flatten to (B*K, L) → backbone → mean-pool tokens → (B*K, d)
          → reshape (B, K, d) → mean-pool episodes → (B, d) → L2-norm
```

This gives a single author-level embedding from K emails, capturing cross-email consistency signals.

---

## Configuration reference

```yaml
encoder:
  name: hf
  model_name_or_path: roberta-base   # any HuggingFace AutoModel
  pooling: mean                       # mean | cls | luar_episode
  freeze_backbone: true               # freeze backbone weights
  max_length: 512                     # truncation length in tokens
  projection_dim: 128                 # optional — project to this size
  lora:                               # optional — omit to disable LoRA
    r: 8
    alpha: 16
    target_modules: [query, value]
    dropout: 0.1
```

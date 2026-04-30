# data/

Data loading, email preprocessing, and P×K batch sampling for contrastive training.

---

## Files

### `base.py` — `BaseDataset` and `EpisodeBatch`

**`EpisodeBatch`** is the canonical batch unit flowing through the training loop:

```python
@dataclass
class EpisodeBatch:
    embeddings: Tensor   # (P*K, d) — filled by Trainer after encoder.encode()
    labels:     Tensor   # (P*K,) integer sender ids (0..P-1, batch-local)
    sender_ids: list[str] # raw sender id strings (for debugging / head.fit())
    texts:      list[str] # raw email texts (for tokenizer)
```

**`BaseDataset`** is the ABC for all email datasets. Subclasses implement:
- `__getitem__(index) → (text, sender_id)` — a single email and its sender
- `sender_ids` property — flat list of sender ids for all items (used by `PKSampler`)

The `episode_collate` classmethod converts a list of `(text, sender_id)` tuples into an `EpisodeBatch` with batch-local integer labels (assigned by first-appearance order).

---

### `enron.py` — `EnronDataset`

Loads the preprocessed Enron corpus from Arrow format. Registered as `"enron"`.

**Prerequisite**: run `scripts/prepare_data.py` to create the Arrow dataset first.

#### Split semantics

| Split | Purpose |
|-------|---------|
| `"train"` | Contrastive training |
| `"validation"` | Hyperparameter search / early stopping |
| `"test"` | Final evaluation only — never used during model selection |

Splits are **sender-disjoint**: no sender appears in more than one split. This simulates the real deployment scenario where the model must recognize senders it has never seen during training.

#### Why Arrow format?

The HuggingFace `datasets` library stores data in Apache Arrow format which is:
- Memory-mapped: data stays on disk until accessed
- O(1) random access by index (no sequential scan)
- Portable across Python/PyTorch versions (no pickle issues)

---

### `preprocessing.py`

Multi-step email cleaning pipeline. Two public entry points:

```python
clean_email_raw(raw: str, config: PreprocessingConfig) → str | None
```
Full pipeline from raw RFC-2822 string. Used by `scripts/prepare_data.py`.

```python
preprocess(body: str, config: PreprocessingConfig) → str | None
```
Pipeline from an already-extracted body. Used by `ScoringPipeline`.

#### Pipeline steps (in order)

| Step | Config flag | What it does |
|------|-------------|-------------|
| Body extraction | always | Parses RFC-2822, prefers text/plain, falls back to HTML |
| Encoding fix | `fix_encoding` | Runs `ftfy` to fix garbled Unicode (â€™ → ') |
| Strip quoted | `strip_quoted` | Removes reply/forward chains (Lotus Notes, Gmail "On...wrote:", "> " lines) |
| Strip signatures | `strip_signatures` | Removes text after `-- ` / `---` / `___` |
| Entity masking | `entity_masking` | Replaces URLs/emails/dates/phones with `[PLACEHOLDER]` tokens |
| Whitespace normalization | always | Collapses tabs, multi-spaces, and 3+ blank lines |
| Truncation | always | Clips to `max_body_chars` |
| Usability check | always | Returns `None` if shorter than `min_body_chars` or >50% placeholders |

#### Why strip replies?

In contrastive stylometry, we want the **author's own words**, not quoted text from other people. A reply email that is 90% quoted text teaches the model nothing about the replying author's style.

#### Entity masking trade-off

Masking high-variance tokens (dates, phone numbers, order IDs) reduces encoder noise from surface-form variation. The cost is losing potentially useful stylometric cues. Disabled by default (`entity_masking: false`).

---

### `samplers.py` — `PKSampler`

Produces **P×K structured batches** for contrastive training.

Each batch contains exactly P senders × K emails each (N = P×K total). This is required by all three contrastive losses — without the guarantee, many batches would have only one email per sender (no positives, no training signal).

```python
PKSampler(
    sender_ids: list[str],  # per-item sender ids
    p: int,                 # senders per batch
    k: int,                 # emails per sender per batch
    drop_last: bool = True,
    seed: int | None = None,
)
```

At each epoch:
1. Shuffles eligible senders (those with >= K emails)
2. Partitions into groups of P senders
3. For each group, samples K emails per sender (without replacement within the batch)
4. Yields a flat list of P×K dataset indices

Senders with fewer than K emails are **silently excluded** from all batches.

---

## Data flow

```
Raw .msg files
      │
      ▼
scripts/prepare_data.py
      │
      ▼  preprocessing + sender-disjoint split
      ▼
Arrow dataset (data/processed/enron/)
      │
      ▼
EnronDataset.__getitem__()  →  (text, sender_id)
      │
      ▼
PKSampler.__iter__()        →  P×K index lists
      │
      ▼
DataLoader(collate_fn=episode_collate)  →  EpisodeBatch
      │
      ▼
Trainer._train_epoch()
```

---

## Configuration reference

```yaml
data:
  dataset: enron
  data_dir: data/raw/enron          # raw .msg files
  processed_dir: data/processed/enron  # Arrow output from prepare_data.py
  train_senders: 100
  min_emails_per_sender: 25
  emails_per_sender_k: 16           # K for PKSampler
  val_split: 0.1
  test_split: 0.1
  preprocessing:
    strip_quoted: true
    strip_signatures: true
    entity_masking: false
    fix_encoding: true
    min_body_chars: 50
    max_body_chars: 4000
```

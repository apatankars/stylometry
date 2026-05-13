# Model Summary — Email Fraud Detection

A reference for how the model works end-to-end, from raw emails to a fraud score.

---

## The Goal

Given a company's email archive, build a per-sender style fingerprint. At runtime, score an incoming email against the claimed sender's fingerprint. A low score means the writing style doesn't match — flag as suspicious.

The system never retrains when deployed. It just builds profiles from historical emails and scores new ones.

---

## Part 1 — Batch Construction (P, K, n_syn)

### Plain PKSampler

Three numbers control what goes into each training batch:

- **P** — how many distinct senders per batch
- **K** — how many emails per sender per batch
- **Batch size** = P × K total emails

Example: P=8, K=4 → 32 emails per batch.

The sampler lays them out as K consecutive rows per sender — all of sender 0's emails come first, then all of sender 1's, etc.:

```
Row  0: alice email 1  ← sender 0
Row  1: alice email 2  ← sender 0
Row  2: alice email 3  ← sender 0
Row  3: alice email 4  ← sender 0
Row  4: bob   email 1  ← sender 1
Row  5: bob   email 2  ← sender 1
...
Row 28: hal   email 1  ← sender 7
...
Row 31: hal   email 4  ← sender 7
```

This layout is not cosmetic — the LUAR reshape depends on it (see Part 2).

### SyntheticBalancedSampler

Adds one extra parameter:

- **n_syn** — number of (real sender, synthetic sender) **pairs** to include per batch

Each pair occupies **2** of the P slots: one for the real sender, one for the LLM imitation of that same sender. The remaining `P - 2*n_syn` slots are filled with ordinary real senders that have no synthetic counterpart in this batch.

Example: P=16, K=4, n_syn=2

```
Slot  0  alice@enron.com        ← real sender  ┐ pair 1
Slot  1  alice@enron.com__syn   ← LLM imitation┘
Slot  2  bob@enron.com          ← real sender  ┐ pair 2
Slot  3  bob@enron.com__syn     ← LLM imitation┘
Slot  4  carol@enron.com        ← unpaired real
Slot  5  dave@enron.com         ← unpaired real
...
Slot 15  pete@enron.com         ← unpaired real
```

Counts:
- Synthetic senders: n_syn = 2
- Real "paired" senders (the originals): also n_syn = 2
- Unpaired real senders: P - 2*n_syn = 16 - 4 = 12
- Total: P = 16

Batch size = 16 × 4 = 64 emails total.

---

## Part 2 — LUAR Episode Pooling

### What LUAR does differently from RoBERTa

Most encoders (RoBERTa, MPNet) take **one email** in and produce **one embedding** out. They were pretrained with masked language modeling, so their embeddings reflect the semantic topic of the text.

LUAR was pretrained specifically for authorship. It takes **K emails from one author** as a group (an "episode") and produces **one embedding** for that episode. The K-email window gives it more stylistic signal than any single email alone.

Crucially: each episode must be from **one sender only**. LUAR never sees emails from multiple senders mixed together in a single episode.

### How LUAR processes a multi-sender batch

The encoder receives the full batch as a flat `(P×K, L)` tensor. It immediately reshapes it into `(P × K/episode_k, episode_k, L)` — a stack of independent single-sender episodes.

This works because PKSampler places K consecutive rows per sender. Slicing into chunks of `episode_k` always stays within one sender.

Concrete example: P=4, K=8, episode_k=4

```
Flat input (32 rows × L tokens):
  rows  0-7  →  alice's 8 emails
  rows  8-15 →  bob's   8 emails
  rows 16-23 →  carol's 8 emails
  rows 24-31 →  dave's  8 emails

After .view(8, 4, L)  →  8 episodes of 4 emails:
  Episode 0: rows  0-3  (alice emails 1-4)  ← single-sender ✓
  Episode 1: rows  4-7  (alice emails 5-8)  ← single-sender ✓
  Episode 2: rows  8-11 (bob   emails 1-4)  ← single-sender ✓
  Episode 3: rows 12-15 (bob   emails 5-8)  ← single-sender ✓
  Episode 4: rows 16-19 (carol emails 1-4)  ← single-sender ✓
  Episode 5: rows 20-23 (carol emails 5-8)  ← single-sender ✓
  Episode 6: rows 24-27 (dave  emails 1-4)  ← single-sender ✓
  Episode 7: rows 28-31 (dave  emails 5-8)  ← single-sender ✓
```

LUAR processes each of the 8 episodes **independently** in parallel (the batch dimension is just GPU parallelism — LUAR has no cross-episode attention). Output: `(8, 512)` — 8 style embeddings.

Each sender now has `K / episode_k = 2` embeddings in the batch. Those 2 embeddings are that sender's positives for the loss.

### Hard constraint: K must be a multiple of episode_k, and K/episode_k ≥ 2

If `K = episode_k`, each sender produces only 1 embedding → no positives → SupCon is undefined. You need at least 2 episodes per sender in the loss batch.

---

## Part 3 — Positives and Negatives

After encoding, the batch contains `N = P × K/episode_k` embeddings (or just `P × K` for non-LUAR). Each embedding has a label = integer sender ID.

**Positive pair**: two embeddings with the **same label** (same `sender_id`). In the example above with P=4 senders and K/episode_k=2, each sender has exactly 2 embeddings → exactly 1 positive pair per sender.

**Negative pair**: any two embeddings with **different labels**.

With `SyntheticBalancedSampler`, crucially:

- `alice@enron.com` → label 0
- `alice@enron.com__syn` → label 1

They have **different labels**, so they are **negatives for each other**, even though the synthetic emails were written to imitate Alice. The contrastive loss is forced to push real-Alice and synthetic-Alice apart in embedding space — which means the model must find stylistic features that the LLM couldn't replicate.

---

## Part 4 — SupCon Loss

For each embedding `i` (the anchor), SupCon computes:

```
L_i = - (1 / |positives for i|)
        × Σ over each positive p:
            log [ sim(i, p) / Σ over all j≠i: sim(i, j) ]

where sim(i, j) = exp( cosine_similarity(i, j) / τ )
```

In plain terms:

- The **numerator** is the similarity to a positive (we want this high)
- The **denominator** is the sum of similarities to everything else in the batch (we want this small relative to the numerator)
- The **log** turns this into a log-probability: "how likely is a randomly picked neighbor to be a positive?"
- We **average** over all positives for anchor i, then **negate** (because we minimize loss = we maximize this probability)
- **τ (temperature)** — typically 0.1. Lower temperature sharpens the softmax: hard negatives (close embeddings from different senders) receive a much stronger gradient signal than easy ones. This is what makes the loss care about synthetic negatives.

The gradients:
- Pull same-sender embeddings together
- Push different-sender embeddings apart, with force proportional to how similar they already are
- Because synthetic emails are stylistically similar to the real sender, they are the hardest negatives and receive the largest push — exactly the pressure needed for fraud detection

---

## Part 5 — Training Loop

```
For each epoch:
  For each batch (from SyntheticBalancedSampler):

    1. episode_collate() assembles EpisodeBatch(texts, labels)
       - texts: flat list of P×K email strings
       - labels: integer sender IDs, P×K long

    2. tokenize(texts)
       → (P×K, max_length) input_ids + attention_mask

    3. encode(input_ids, attention_mask)
       → reshape to (P×K/episode_k, episode_k, max_length)  [LUAR path]
       → LUAR forward: each episode → one style embedding
       → output: (P×K/episode_k, 512)  L2-normalized

    4. stride labels: labels[::episode_k]
       → (P×K/episode_k,)  one label per episode

    5. SupConLoss(embeddings, strided_labels)
       → scalar loss

    6. loss.backward()
       → gradients flow through L2-norm, LUAR episode attention,
          into LoRA adapter matrices only (backbone frozen)

    7. clip gradients at 1.0
       optimizer.step()  (only LoRA params updated)
       scheduler.step()  (cosine LR, per batch)
```

After all epochs: run `head.fit()` to build sender profiles from reference emails.

---

## Part 6 — Inference (Deployment)

### Enrollment (done once per sender, offline)

```
For each sender in the company:
  Collect their known-genuine historical emails
  → preprocess each email (strip quotes/sigs, mask entities)
  → encode each email through the trained LUAR encoder
  → compute centroid = mean(all embeddings)
  → compute spread   = mean(1 - cosine_similarity(each email, centroid))
  → store {centroid, spread, k=email count}

Save profiles to disk.
```

`spread` measures how consistently this sender writes. A sender who always writes the same way has low spread; a sender who writes very differently depending on context has high spread.

### Scoring (at runtime, per incoming email)

```
Input: email_text, claimed_sender_id

1. preprocess(email_text)
2. tokenize → encode → (1, 512) L2-normalized embedding

3. Load profile for claimed_sender_id:
     centroid (512,), spread (float), k (int)

4. cosine_distance = 1 - cosine_similarity(query, centroid)

5. z = cosine_distance / spread
      "how many standard deviations from this sender's typical style?"

6. score = max(0.0, 1.0 - z / 3.0)
     z = 0 → score = 1.0  (query is exactly at the centroid)
     z = 1 → score = 0.67 (1 std dev away, slightly unusual)
     z = 3 → score = 0.0  (3 std devs away, very atypical)
     z > 3 → score = 0.0  (clamped)

7. tier = confidence based on k:
     k < 5     → "low"       → abstain = True  (not enough history)
     k 5-9     → "medium"
     k 10-24   → "high"
     k ≥ 25    → "very_high"
```

**Why this works**: the trained encoder arranged embedding space so same-sender emails cluster together and different-sender emails spread apart. A genuine email lands near the centroid. An impersonation (human or LLM) lands farther away because the writing style doesn't match the learned pattern.

---

## Configuration Quick Reference

| Parameter | Controls |
|---|---|
| `P` | Senders per batch. More P = more diverse negatives per step |
| `K` | Emails per sender per batch. Must be ≥ 2 × episode_k so each sender gets ≥ 2 episodes |
| `episode_k` | How many emails LUAR pools into one style embedding. LUAR processes each episode independently |
| `n_syn` | Number of (real, synthetic) pairs per batch. Each pair uses 2 of the P slots. Remaining P - 2×n_syn slots are unpaired real senders |
| `τ (temperature)` | SupCon sharpness. 0.1 is standard. Lower = harder negatives get stronger gradient |

**Constraint summary**: `K % episode_k == 0` and `K / episode_k >= 2` and `2 * n_syn <= P`

---

## Why LUAR Outperforms RoBERTa

| | RoBERTa | LUAR |
|---|---|---|
| Pretraining task | Predict masked tokens | Distinguish authors from each other |
| What embeddings encode | Semantic topic | Stylometric fingerprint |
| Two emails about the same topic | Close together | Depends on author, not topic |
| Fine-tuning with SupCon | Nudges representations, but wrong inductive bias | Builds on already-stylometric representations |
| val/auroc after 20 epochs | ~0.53 (near random) | ~0.95 |

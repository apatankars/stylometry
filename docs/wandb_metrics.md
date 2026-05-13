# W&B Metrics Reference

Every metric the trainer logs, what it measures, and how to read it. Two categories:

1. **Training metrics** — computed from the loss signal and validation embeddings each epoch.
2. **Inference metrics** (`val/centroid/*`) — computed from a fixed *centroid probe* that mimics deployment: enroll senders by averaging embeddings, then score genuine / impostor / synthetic queries against those centroids.

---

## Vocabulary

These terms appear throughout. Read these first.

### "Genuine"
A query email actually written by the person whose centroid we're scoring it against. Concretely: for each profiled sender we hold out N enrollment emails to build the centroid; the *remaining* real emails from that same sender are **genuine queries**. A well-behaved model assigns these high scores.

### "Other-sender impostor"
An email written by a person who is *not* the claimed sender, drawn from a sender-disjoint split (validation or test) so the encoder has never seen them. Easy negative: different person, different style entirely. A model has to assign these low scores.

### "Synthetic" (hard negative)
An LLM-generated email produced by `scripts/generate_synthetic_emails.py`: the model was prompted with N real emails from a sender and asked to write a new email mimicking that person's voice on an unrelated topic. Same person's claimed style, different content distribution. A model that's only learned topical/lexical shortcuts will score these high; a model that's learned actual stylometric features will score them low. This is the *hard* negative — the case where fraud detection has to actually work.

### "Report"
The model "reports" on a query when it commits a verdict that the email is genuine — concretely, when the head's score crosses a threshold τ. The system can be tuned to report only on its most confident outputs (high τ, fewer reports, higher precision) or to report broadly (low τ, more reports, more leakage). The boss's ">50% / >80% / >95%" framing is asking how the model behaves at three operating points: τ = 0.5, 0.8, 0.95.

### "Centroid"
The mean of L2-normalised embeddings of a sender's enrollment emails. A new query's score is `1 - z/3` where `z` is the cosine distance from the query to the centroid divided by the sender's intra-class spread (mean cosine distance of enrollment emails to the centroid). Higher = more consistent with the sender's style.

### "Sender-disjoint"
Train, validation, and test splits never share a sender. Critical: a model that has seen Alice's emails at training time has an unfair advantage at test time. Sender-disjoint splits simulate the real deployment scenario where the model must recognise *new* senders it has never seen.

---

## Training metrics

Logged every epoch.

| Metric | What it measures |
|---|---|
| `epoch` | Current training epoch (1-indexed). |
| `train/loss` | Mean loss over all training batches in this epoch. SupCon (Khosla et al. 2020) or batch-hard Triplet (Hermans et al. 2017), depending on `loss.name`. Lower = the encoder is pulling same-sender embeddings together and pushing different-sender embeddings apart. |
| `train/lr` | Current learning rate from the scheduler. With cosine + warmup, this ramps from ~0 to `training.lr` over `warmup_steps`, then decays toward 0 across the rest of training. Useful for diagnosing convergence problems. |

## Validation metrics

Computed each epoch from a held-out PKSampler batch over the validation split. The validation set is sender-disjoint from train, so these numbers measure how well the encoder generalises stylometric structure to people it has never seen.

| Metric | What it measures |
|---|---|
| `val/loss` | Same loss function as training, evaluated on the val split. Used by the `checkpoint_best.pt` selector. |
| `val/auroc` | Pairwise authorship-verification AUROC. For every (i, j) pair in the val batch (upper triangle of the cosine-similarity matrix), score = cosine similarity, label = 1 if same sender, 0 if different. AUROC is the probability that a randomly chosen same-sender pair scores higher than a randomly chosen different-sender pair. **The headline val number.** 0.5 = random; 1.0 = perfect ranking. |
| `val/knn_acc` | Leave-one-out 1-nearest-neighbour accuracy in the val batch. For each email, find the most similar other email; the prediction is "right" if they share a sender. Sensitive to local geometry — an embedding space can have great AUROC but poor 1-NN if there are dense impostor clusters. |
| `val/macro_f1` | Macro-averaged F1 of the same 1-NN predictions. Per-sender F1 is computed (treating each sender as a one-vs-rest binary task), then averaged with equal weight per sender. Macro (not micro) so dominant senders don't drown out small ones — important here because Enron has highly skewed email counts per sender. |

### What `val/macro_f1` is computing in detail

For a val batch with K distinct senders:
1. For each query, predict the sender of its nearest neighbour (excluding itself).
2. For each sender s ∈ {1...K}: treat predictions as a binary task ("is sender s or not"), compute precision_s, recall_s, then F1_s = 2·P·R/(P+R).
3. macro_f1 = mean(F1_1 ... F1_K).

Equal weight per sender. A sender with 4 val emails contributes the same to the average as a sender with 40. AUROC weights pairs (so big senders dominate) — macro F1 weights identities.

---

## Centroid-probe metrics (`val/centroid/*`)

Same probe set every epoch (fixed at training start so only the encoder weights change). Built from train senders profiled with held-out enrollment emails; queries pulled from val (impostors) and the synthetic dataset (hard negatives).

### Score statistics

| Metric | What it measures |
|---|---|
| `val/centroid/score_genuine` | Mean head score for genuine queries. Should be high (target ≥ 0.7 for a well-trained model). |
| `val/centroid/score_other` | Mean head score for other-sender impostors. Should be low (target ≤ 0.3). |
| `val/centroid/score_synthetic` | Mean head score for LLM-generated synthetic emails scored against their imitation target's centroid. Should be lower than genuine. **If this is close to score_genuine, the encoder is being fooled by topical mimicry.** |
| `val/centroid/gap_other` | `score_genuine - score_other`. Margin against easy negatives. |
| `val/centroid/gap_synthetic` | `score_genuine - score_synthetic`. Margin against hard negatives. Smaller = harder. |
| `val/centroid/synthetic_harder` | `gap_other - gap_synthetic`. **Positive ⇒ synthetics are harder than other-sender (the augmentation is doing real work). Negative ⇒ synthetics are *easier* than impostors, which suggests the LLM is leaving artefacts the model picks up on rather than learning stylometry.** |
| `val/centroid/n_genuine`, `n_other`, `n_synthetic` | Count of queries in each pool. |

### Discrimination AUROCs

| Metric | What it measures |
|---|---|
| `val/centroid/auroc_genuine_vs_other` | AUROC distinguishing genuine from other-sender impostors using the head score. Easy task — should approach 1.0 quickly during training. |
| `val/centroid/auroc_genuine_vs_synthetic` | AUROC distinguishing genuine from synthetic. **The hard-negative number.** This is the metric that says whether your model has learned style or topic. |
| `val/centroid/auroc_genuine_vs_all` | AUROC against the pooled impostor set (other ∪ synthetic). Production-relevant aggregate. |

### Operating-point reports (the boss's >50% / >80% / >95% framing)

For each threshold τ ∈ {0.5, 0.8, 0.95}, the model "reports" (predicts genuine) iff score > τ. `@τ` notation uses the literal value (`@0.5`, `@0.8`, `@0.95`). All metrics are computed over genuine + (other ∪ synthetic).

| Metric | Definition | What it tells you |
|---|---|---|
| `val/centroid/report_rate@τ` | (# queries with score > τ) / N | What fraction of inputs the system commits a verdict on. Higher τ ⇒ lower coverage. |
| `val/centroid/precision@τ` | TP / (TP + FP) | Of the reports issued, fraction that are correct. **The trustworthiness of a verdict at this band.** |
| `val/centroid/recall@τ` | TP / (TP + FN) = (genuine > τ) / N_genuine | Fraction of genuine emails the system catches. |
| `val/centroid/accuracy@τ` | (TP + TN) / N | Overall correctness treating "score > τ" as the genuine prediction. |
| `val/centroid/fpr_other@τ` | (other > τ) / N_other | Fraction of other-sender impostors that get past τ. |
| `val/centroid/fpr_synthetic@τ` | (synthetic > τ) / N_syn | **Fraction of LLM imitations that fool the model at this confidence band.** Watch this at τ=0.95 — non-trivial values mean high-confidence verdicts can still be deceived. |
| `val/centroid/fpr_overall@τ` | (impostors > τ) / N_imp | Pooled FPR across both impostor types. |
| `val/centroid/genuine_above@τ` | Same as `recall@τ` (kept for readability). | |

How to read this together:

```
At τ = 0.95 we want:
    precision@0.95            close to 1.0   (verdict almost always right)
    fpr_synthetic@0.95        close to 0.0   (synthetics don't slip through)
    recall@0.95               as high as possible (we still catch genuine)
```

If `precision@0.95` is high but `fpr_synthetic@0.95` is also non-trivial, your high-confidence reports look good *only because the impostor pool is dominated by easy other-sender cases*. That's the gap V4 (synthetic-augmented training) is meant to close.

### Selective-classification coverage

For each accuracy target T ∈ {0.5, 0.8, 0.95}, sort queries by confidence (|score − 0.5|) descending. Compute running accuracy as we accept progressively lower-confidence predictions. Find the largest prefix length whose running accuracy still ≥ T.

| Metric | Definition |
|---|---|
| `val/centroid/coverage_at_acc@T` | Largest fraction of the query stream we can "report on" while keeping accuracy ≥ T. |

Read as: "what coverage can we offer at this accuracy guarantee?". `coverage_at_acc@0.95 = 0.40` means: if we only commit verdicts on the most confident 40% of queries, those verdicts will be ≥95% accurate.

This is the dual view to the threshold metrics. Threshold metrics fix τ on the raw score and report what falls out; coverage-at-accuracy fixes the accuracy goal and reports the coverage it implies.

---

## Test-set PAN metrics (every 5 epochs, prefixed `test/`)

The trainer scores `data/processed/enron/test_pairs.jsonl` inline (no subprocess) and logs the four PAN authorship-verification shared-task metrics into the same W&B run.

| Metric | What it measures |
|---|---|
| `test/auc` | ROC-AUC over verification pairs (each row is two emails + a "same author" label). |
| `test/c_at_1` | Peñas & Rodrigo (2011) accuracy-with-abstention. Rewards correct decisions, penalises wrong ones, allows abstention without penalty. |
| `test/f_05_u` | PAN 2020 primary metric. F-score weighted toward precision (β=0.5), with abstained pairs distributed across true/false predictions. |
| `test/eer` | Equal Error Rate — the error rate at the threshold where FPR = FNR. Threshold-free, lower = better. |

These run on the actual sender-disjoint test pairs file and are the closest thing to a paper-grade number the trainer logs in-flight.

---

## Putting it together — what to watch during a run

1. **Convergence**: `train/loss` decreasing, `val/loss` plateauing (not increasing → overfitting).
2. **Style learning vs. topic shortcuts**: `val/centroid/auroc_genuine_vs_other` rising toward 1.0 fast, then `val/centroid/auroc_genuine_vs_synthetic` rising more slowly. If the latter never moves, the model has learned topical/lexical shortcuts only.
3. **Hard-negative effect (V4 vs V2)**: `val/centroid/synthetic_harder` should be positive and growing in V4 relative to V2.
4. **Operating points**: `val/centroid/precision@0.95` and `val/centroid/fpr_synthetic@0.95` together — high-confidence verdicts must be both precise and synthetic-resistant.
5. **Headline numbers for reporting**: `test/auc`, `test/eer`, `val/centroid/auroc_genuine_vs_all`, `val/centroid/coverage_at_acc@0.95`.

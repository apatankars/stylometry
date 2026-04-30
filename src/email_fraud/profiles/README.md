# profiles/

In-memory per-sender embedding profile store with EWMA updates.

---

## Files

### `store.py` — `SenderProfileStore`

Maintains a compact representation of each known sender's "normal" writing style as a centroid in embedding space.

#### Profile structure

```python
{
    sender_id: {
        "centroid": np.ndarray (d,),  # EWMA-averaged embedding, L2-normalized
        "spread":   float,            # EWMA of cosine distance from new emails to centroid
        "k":        int,              # total emails incorporated
        "metadata": dict,             # arbitrary extras (e.g. last_seen timestamp)
    }
}
```

#### EWMA centroid update

```python
centroid_new = (1 - α) * centroid_old + α * embedding
centroid_new = centroid_new / ||centroid_new||   # re-normalize to unit sphere
spread_new   = (1 - α) * spread_old + α * cos_dist(embedding, centroid_new)
```

Default `α = 0.1`: each new email shifts the centroid by 10% toward the new embedding. This makes the profile robust to individual noisy emails while still adapting over time.

**Difference from `PrototypicalHead.fit()`**: The head uses equal-weight running average (all past emails have equal weight). The store uses EWMA (recent emails have slightly higher weight). Use the store when you want the profile to drift with the sender's evolving style; use the head for a frozen reference profile.

---

#### Key methods

| Method | Description |
|--------|-------------|
| `upsert(sender_id, embedding, metadata)` | Insert first email or EWMA-update existing profile |
| `get_profile(sender_id)` | Return profile dict or `None` if unknown |
| `confidence_tier(sender_id)` | Return tier string based on `k` |
| `mahalanobis_score(sender_id, query)` | **Stub** — returns 0.0; TODO |
| `save(path)` | Serialize all profiles to JSON (centroids as lists) |
| `load(path)` | Restore profiles from JSON |
| `__len__()` | Number of known senders |
| `__contains__(sender_id)` | Check if sender has a profile |

#### Persistence format

JSON file:
```json
{
  "user@example.com": {
    "centroid": [0.12, -0.34, ...],
    "spread": 0.042,
    "k": 47,
    "metadata": {}
  }
}
```

#### Production note

The current dict-backed store is single-process and in-memory. For multi-worker deployments, the TODO comment suggests replacing it with Postgres `pgvector` for shared state and persistence across restarts.

---

## Relationship to `PrototypicalHead`

`SenderProfileStore` and `PrototypicalHead` both maintain per-sender centroids, but serve different roles:

| | `SenderProfileStore` | `PrototypicalHead` |
|--|---------------------|-------------------|
| Update rule | EWMA (recency bias) | Equal-weight running average |
| Storage format | numpy arrays (JSON-serializable) | PyTorch tensors (pickle) |
| Used by | `ScoringPipeline` (inference) | `Trainer` (training + inference) |
| Tier lookup | ✓ | ✓ |
| Mahalanobis | stub | stub |

In production, you would likely use either one or the other, not both. The pipeline currently uses the head for scoring and the store for optional online updates.

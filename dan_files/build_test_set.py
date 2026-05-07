"""
Outputs:
  test_set.jsonl       — test examples consumed by evaluate.py
  test_set_ids.txt     — message_ids of all held-out emails; pass to load_enron()
                         via --exclude-ids so neither model trains on these.

Test set design
---------------
For each Enron author a stratified 15% of their emails are held out as the
test pool. From that pool, examples are generated at four context sizes
(n_context ∈ {1, 3, 5, 10}) and three label types:

  positive   — reference and query both from the same author          (label 1)
  random_neg — reference from author A, query from a random author B  (label 0)
  hard_neg   — reference from author A, query from the most TF-IDF-
               similar author B (hardest stylistic impersonation)      (label 0)

N_QUERIES_PER_CELL examples are generated per (author, n_context, type)
combination, giving a balanced, stratified test set.

Each example record
-------------------
{
  "example_id":       "<author>_<uid>_<type>_ctx<n>",
  "reference_author": str,        # whose style the reference emails represent
  "true_author":      str,        # actual author of query_email
  "reference_emails": [str, ...], # n_context cleaned email bodies
  "query_email":      str,        # the email to classify
  "label":            0 | 1,
  "n_context":        int,
  "negative_type":    null | "random" | "hard"
}
"""

import json
import pathlib
import random
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_cleaning import load_enron

# Config

ENRON_ROOT        = pathlib.Path("C:/Users/danil/Documents/enr_data")
OUT_JSONL         = pathlib.Path("/workspace/test_set.jsonl")
OUT_IDS           = pathlib.Path("/workspace/test_set_ids.txt")
TEST_FRAC         = 0.15
N_CONTEXT_SIZES   = [1, 3, 5, 10]
N_QUERIES_PER_CELL = 5          # examples per (author, n_context, type)
MIN_TEST_POOL     = 12          # skip authors whose test pool is too small
SEED              = 0           # fixed forever

# Hard-negative map: for each author, find most TF-IDF-similar other author

def _build_hard_neg_map(df: pd.DataFrame) -> dict[str, str]:
    author_corpora = {
        author: " ".join(group["body"].tolist())
        for author, group in df.groupby("author")
    }
    authors = list(author_corpora.keys())
    corpora = [author_corpora[a] for a in authors]

    vec    = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform(corpora)
    sims   = cosine_similarity(matrix)

    hard_neg = {}
    for i, author in enumerate(authors):
        row      = sims[i].copy()
        row[i]   = -1.0          # exclude self
        hard_neg[author] = authors[int(np.argmax(row))]
    return hard_neg

# Main

def build(enron_root: pathlib.Path = ENRON_ROOT) -> None:
    rng = random.Random(SEED)

    print("Loading Enron…")
    df = load_enron(root=enron_root)
    print(f"  {len(df):,} emails  {df['author'].nunique()} authors")

    print("Computing TF-IDF hard-negative map…")
    hard_neg_map = _build_hard_neg_map(df)

    # Stratified hold-out: 15% per author → test pool
    print(f"Splitting {TEST_FRAC:.0%} per author into test pool…")
    test_frames, train_frames = [], []
    for author, group in df.groupby("author"):
        group  = group.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_test = max(1, int(len(group) * TEST_FRAC))
        test_frames.append(group.iloc[:n_test])
        train_frames.append(group.iloc[n_test:])

    test_df  = pd.concat(test_frames).reset_index(drop=True)
    print(f"  Test pool : {len(test_df):,} emails")

    # Collect all test-pool message IDs for training exclusion
    test_ids = set(test_df["message_id"].dropna().unique())

    # Index: author → list of cleaned bodies in the test pool
    author_pool: dict[str, list[str]] = {
        author: group["body"].tolist()
        for author, group in test_df.groupby("author")
    }
    authors = [a for a, pool in author_pool.items() if len(pool) >= MIN_TEST_POOL]
    print(f"  Authors with ≥{MIN_TEST_POOL} test emails: {len(authors)}")

    examples = []
    uid      = 0

    for author in authors:
        pool       = author_pool[author]
        rand_neg   = rng.choice([a for a in authors if a != author])
        hard_neg   = hard_neg_map.get(author, rand_neg)
        if hard_neg not in author_pool or hard_neg == author:
            hard_neg = rand_neg

        for n_ctx in N_CONTEXT_SIZES:
            # Need at least n_ctx + 1 emails to have a disjoint query
            if len(pool) < n_ctx + 1:
                continue

            for _ in range(N_QUERIES_PER_CELL):
                # positive
                sample   = rng.sample(pool, n_ctx + 1)
                ref, qry = sample[:n_ctx], sample[n_ctx]
                examples.append({
                    "example_id":       f"{author}_{uid:05d}_positive_ctx{n_ctx}",
                    "reference_author": author,
                    "true_author":      author,
                    "reference_emails": ref,
                    "query_email":      qry,
                    "label":            1,
                    "n_context":        n_ctx,
                    "negative_type":    None,
                })
                uid += 1

                # random negative
                rand_pool = author_pool.get(rand_neg, [])
                if rand_pool:
                    ref   = rng.sample(pool, min(n_ctx, len(pool)))
                    qry   = rng.choice(rand_pool)
                    examples.append({
                        "example_id":       f"{author}_{uid:05d}_random_neg_ctx{n_ctx}",
                        "reference_author": author,
                        "true_author":      rand_neg,
                        "reference_emails": ref,
                        "query_email":      qry,
                        "label":            0,
                        "n_context":        len(ref),
                        "negative_type":    "random",
                    })
                    uid += 1

                # hard negative
                hard_pool = author_pool.get(hard_neg, [])
                if hard_pool:
                    ref   = rng.sample(pool, min(n_ctx, len(pool)))
                    qry   = rng.choice(hard_pool)
                    examples.append({
                        "example_id":       f"{author}_{uid:05d}_hard_neg_ctx{n_ctx}",
                        "reference_author": author,
                        "true_author":      hard_neg,
                        "reference_emails": ref,
                        "query_email":      qry,
                        "label":            0,
                        "n_context":        len(ref),
                        "negative_type":    "hard",
                    })
                    uid += 1

    rng.shuffle(examples)
    print(f"\nGenerated {len(examples):,} test examples")
    print(f"  Positive : {sum(1 for e in examples if e['label'] == 1):,}")
    print(f"  Random   : {sum(1 for e in examples if e['negative_type'] == 'random'):,}")
    print(f"  Hard     : {sum(1 for e in examples if e['negative_type'] == 'hard'):,}")

    OUT_JSONL.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in examples),
        encoding="utf-8",
    )
    OUT_IDS.write_text("\n".join(sorted(test_ids)), encoding="utf-8")

    print(f"\nSaved → {OUT_JSONL}  ({OUT_JSONL.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved → {OUT_IDS}  ({len(test_ids):,} message IDs)")
    print("\nNext steps:")
    print("  1. Pass --exclude-ids test_set_ids.txt when training both models")
    print("  2. Run evaluate.py after training to compare approaches")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--enron-root", default=str(ENRON_ROOT))
    args = p.parse_args()
    build(pathlib.Path(args.enron_root))

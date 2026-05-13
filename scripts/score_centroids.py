"""Centroid-based inference evaluation, mirroring deployment.

Deployment story
----------------
At inference time the encoder is **frozen**.  A user enrolls each known sender
by averaging the embeddings of a handful of their emails into a centroid; new
emails are scored by cosine distance to that centroid (z-score normalised by
the sender's intra-class spread).  No further training happens.

This script measures exactly that pipeline against three discrimination tasks:

    1. genuine  vs other-sender  — easy:  different person, different style
    2. genuine  vs synthetic     — hard:  same person's style, LLM-imitated
    3. genuine  vs (other ∪ syn) — pooled impostors

For each profiled sender we hold out N enrollment emails (centroid) and use the
rest as genuine queries.  Impostor queries are drawn from sender-disjoint test
emails (people the encoder was not trained on).  Synthetic queries come from
data/synthetic/enron_synthetic, scored against the source-sender's centroid.

Usage::

    python scripts/score_centroids.py \\
        --run runs/v2_luar_lora/2026-05-02_21-59-01

    python scripts/score_centroids.py \\
        --run runs/v4_luar_lora_syn/<ts> \\
        --include-synthetic data/synthetic/enron_synthetic \\
        --wandb
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.encoders  # noqa: F401
import email_fraud.heads     # noqa: F401
import email_fraud.losses    # noqa: F401
from email_fraud.config import load_config
from email_fraud.registry import resolve
from email_fraud.scoring.centroid_probe import CentroidProbe
from email_fraud.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _abs(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_PROJECT_ROOT / pp).resolve()


def _resolve_checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        cand = run_dir / name
        if cand.exists():
            return cand
    epoch_pts = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
    if not epoch_pts:
        raise FileNotFoundError(f"No .pt checkpoint found in {run_dir}")
    return epoch_pts[-1]


def _load_split(processed_dir: Path, split: str) -> tuple[list[str], list[str]]:
    from datasets import load_from_disk

    ds_dict = load_from_disk(str(processed_dir))
    if split not in ds_dict:
        raise ValueError(f"Split '{split}' not in {processed_dir}")
    ds = ds_dict[split]
    return list(ds["text"]), list(ds["sender_id"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Centroid-style inference eval (genuine / other / synthetic).",
    )
    p.add_argument("--run", required=True, metavar="DIR")
    p.add_argument("--config", default=None)
    p.add_argument("--profile-split", default="train",
                   help="Split whose senders we profile (default: train — these "
                        "are the people the encoder learned).")
    p.add_argument("--impostor-split", default="test",
                   help="Split providing other-sender impostor queries "
                        "(default: test — sender-disjoint from train).")
    p.add_argument("--include-synthetic", default=None, metavar="ARROW_DIR",
                   help="Path to synthetic Arrow dataset for hard negatives.")
    p.add_argument("--n-profile-senders", type=int, default=50)
    p.add_argument("--n-enroll-per-sender", type=int, default=8)
    p.add_argument("--n-query-per-sender", type=int, default=8)
    p.add_argument("--n-other-queries", type=int, default=400)
    p.add_argument("--n-synthetic-queries", type=int, default=400)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    run_dir = _abs(args.run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    checkpoint = _resolve_checkpoint(run_dir)
    config_path = _abs(args.config) if args.config else (run_dir / "config.yaml")
    cfg = load_config(str(config_path))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Config:     %s", config_path)
    logger.info("Device:     %s", device)

    EncoderClass = resolve("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = torch.load(str(checkpoint), map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    encoder.to(device)
    logger.info("Loaded weights from epoch %s", ckpt.get("epoch", "?"))

    processed_dir = _abs(cfg.data.processed_dir)
    profile_texts, profile_senders = _load_split(processed_dir, args.profile_split)
    other_texts, other_senders = _load_split(processed_dir, args.impostor_split)

    syn_texts: list[str] = []
    syn_sources: list[str] = []
    if args.include_synthetic:
        from datasets import load_from_disk
        syn_path = _abs(args.include_synthetic)
        syn_ds = load_from_disk(str(syn_path))
        syn_texts = list(syn_ds["text"])
        syn_sources = list(syn_ds["source_sender_id"])
        logger.info("Loaded %d synthetic emails from %s", len(syn_texts), syn_path)

    probe = CentroidProbe(
        train_texts=profile_texts,
        train_senders=profile_senders,
        other_texts=other_texts,
        other_senders=other_senders,
        synthetic_texts=syn_texts or None,
        synthetic_source_senders=syn_sources or None,
        n_profile_senders=args.n_profile_senders,
        n_enroll_per_sender=args.n_enroll_per_sender,
        n_query_per_sender=args.n_query_per_sender,
        n_other_queries=args.n_other_queries,
        n_synthetic_queries=args.n_synthetic_queries,
        confidence_tiers=cfg.confidence_tiers,
        seed=args.seed,
    )

    metrics = probe.evaluate(encoder, device, batch_size=args.batch_size)

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Run            : {run_dir.parent.name}/{run_dir.name}")
    print(f"  Checkpoint     : {checkpoint.name}  (epoch {ckpt.get('epoch', '?')})")
    print(f"  Profile split  : {args.profile_split}")
    print(f"  Impostor split : {args.impostor_split}")
    print(f"  Synthetic      : {args.include_synthetic or '(none)'}")
    print(sep)
    summary_keys = [
        "auc/genuine_vs_other",
        "auc/genuine_vs_synthetic",
        "auc/genuine_vs_all",
        "score/mean_genuine",
        "score/mean_other",
        "score/mean_synthetic",
        "score/gap_other",
        "score/gap_synthetic",
        "score/synthetic_harder_than_other",
        "probe/n_genuine_queries",
        "probe/n_other_queries",
        "probe/n_synthetic_queries",
    ]
    for key in summary_keys:
        if key in metrics:
            print(f"  {key:<40} {metrics[key]:.4f}")

    # Threshold-band table — the "reports >50%/>80%/>95%" view.
    print(sep)
    print(f"  {'threshold':<12}{'report_rate':>12}{'precision':>11}"
          f"{'recall':>9}{'fpr_other':>11}{'fpr_syn':>10}")
    for tau in (0.5, 0.8, 0.95):
        group = f"threshold_{tau:.2f}"
        rr = metrics.get(f"{group}/report_rate", float("nan"))
        pr = metrics.get(f"{group}/precision", float("nan"))
        rc = metrics.get(f"{group}/recall", float("nan"))
        fo = metrics.get(f"{group}/fpr_other", float("nan"))
        fs = metrics.get(f"{group}/fpr_synthetic", float("nan"))
        print(f"  >{tau:<11.2f}{rr:>12.3f}{pr:>11.3f}{rc:>9.3f}{fo:>11.3f}{fs:>10.3f}")

    # Selective-classification view: coverage @ accuracy target.
    print(sep)
    for target in (0.5, 0.8, 0.95):
        cov = metrics.get(f"coverage/at_acc_{target:.2f}", float("nan"))
        print(f"  coverage @ accuracy ≥ {target:.2f}      {cov:.3f}")
    print(f"{sep}\n")

    if args.wandb:
        import wandb
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=(cfg.wandb.tags or []) + ["centroid-eval"],
            notes=cfg.wandb.notes,
            config={
                "checkpoint": str(checkpoint),
                "profile_split": args.profile_split,
                "impostor_split": args.impostor_split,
                "synthetic": args.include_synthetic,
                "n_profile_senders": args.n_profile_senders,
                "n_enroll_per_sender": args.n_enroll_per_sender,
            },
        )
        wandb.log(metrics)
        wandb.summary.update(metrics)
        run.finish()


if __name__ == "__main__":
    main()

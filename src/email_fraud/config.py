"""Pydantic v2 config schema and YAML loader.

ExperimentConfig is the single source of truth for every run.  Loading merges
configs/base.yaml (defaults) with the experiment file so experiment files only
need to specify what differs from the defaults.

Config composition
------------------
1. configs/base.yaml  — global defaults (never checked in with secrets)
2. configs/<experiment>.yaml — overrides for a specific experiment
3. _deep_merge() — recursively merges so nested keys can be partially overridden

Example: to change only the learning rate, your experiment yaml only needs:
    training:
      lr: 5e-5

All other training fields inherit from base.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation settings for the transformer backbone.

    LoRA adds trainable rank-r matrices to targeted attention layers.
    Only adapter weights are updated; the backbone remains frozen.

    r: rank of the adapter matrices (higher = more capacity, more params).
    alpha: scaling factor; effective lr for adapters is (alpha/r) * lr.
    target_modules: which attention projections to adapt.
      ["query", "value"] is the standard config from the LoRA paper (Hu et al. 2021).
    dropout: regularization on adapter activations.
    """
    model_config = ConfigDict(extra="forbid")

    r: int = 8
    alpha: int = 16
    target_modules: list[str] = Field(default_factory=lambda: ["query", "value"])
    dropout: float = 0.1


class EncoderConfig(BaseModel):
    """Configuration for the text encoder (backbone + pooling + optional projection).

    name: registry key for the encoder class ("hf" → HFEncoder).
    model_name_or_path: HuggingFace model id or local path to a checkpoint.
    pooling: token aggregation strategy — "mean", "cls", or "luar_episode".
    lora: if set, wrap backbone with LoRA adapters (cheap fine-tuning).
    freeze_backbone: if True and lora is None, backbone is a fixed feature extractor;
                     only a projection head (if any) is trained.
    max_length: max tokens per email; longer inputs are silently truncated.
    projection_dim: if set, add a linear layer backbone_dim → projection_dim.
                    Useful for controlling embedding size independently of backbone.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "hf"
    model_name_or_path: str = "roberta-base"
    pooling: str = "mean"
    lora: LoRAConfig | None = None
    freeze_backbone: bool = True
    max_length: int = 512
    projection_dim: int | None = None  # trainable linear projection on top of backbone
    episode_k: int | None = None  # emails per LUAR episode; required when pooling='luar_episode'


class LossConfig(BaseModel):
    """Configuration for the contrastive loss function.

    name: registry key — "supcon", "triplet", or "contrastive".
    temperature: for SupConLoss; lower = sharper softmax (harder negatives dominate).
    margin: for TripletLoss and ContrastiveLoss; minimum separation for negative pairs.
    mining: pair/triplet mining strategy:
      "batch_hard" (triplet) — hardest positive + hardest negative per anchor.
      "all" (triplet/contrastive) — all valid (anchor, pos, neg) combinations.
      "semi_hard" (contrastive) — negatives outside the positive cluster but inside margin.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "supcon"
    temperature: float = 0.1
    margin: float = 0.3
    mining: str = "batch_hard"


class HeadConfig(BaseModel):
    """Configuration for the anomaly-scoring head.

    name: registry key — "prototypical" or "cross_encoder".
    distance: metric for centroid-to-query comparison ("cosine" is the only live option).
    shrinkage: covariance shrinkage for Mahalanobis scoring (stub, not yet implemented).
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "prototypical"
    distance: str = "cosine"
    shrinkage: str = "ledoit_wolf"


class PreprocessingConfig(BaseModel):
    """Controls which cleaning steps are applied to email bodies.

    strip_quoted: remove reply/forward chains (keeps only the newest message).
    strip_signatures: cut text after signature separators (-- / --- / ___).
    entity_masking: replace URLs, emails, dates, phones with [PLACEHOLDER] tokens.
                    Reduces encoder noise but removes potentially useful stylometric cues.
    fix_encoding: run ftfy to fix garbled Unicode common in old Enron data.
    min_body_chars: drop emails shorter than this after cleaning (50 ≈ 10 words).
    max_body_chars: truncate bodies longer than this (4000 chars ≈ 800 tokens).
    """
    model_config = ConfigDict(extra="forbid")

    strip_quoted: bool = True
    strip_signatures: bool = True
    entity_masking: bool = False
    fix_encoding: bool = True          # run ftfy to fix garbled unicode/encoding artifacts
    min_body_chars: int = 50           # drop emails shorter than this after cleaning
    max_body_chars: int = 4000         # truncate bodies longer than this


class DataConfig(BaseModel):
    """Dataset paths and sampling parameters.

    dataset: registry key for the dataset class ("enron" → EnronDataset).
    data_dir: path to raw .msg files (consumed by scripts/prepare_data.py only).
    processed_dir: path to Arrow-format processed dataset (consumed by EnronDataset).
    train_senders: number of senders in the training split.
    min_emails_per_sender: senders with fewer are excluded from all splits.
    emails_per_sender_k: target K for PKSampler (emails per sender per batch).
    val_split / test_split: fraction of eligible senders reserved for val/test.
    preprocessing: nested config controlling email cleaning.
    """
    model_config = ConfigDict(extra="forbid")

    dataset: str = "enron"
    data_dir: str = "data/raw/enron"
    processed_dir: str = "data/processed/enron"
    train_senders: int = 100
    min_emails_per_sender: int = 25
    emails_per_sender_k: int = 16
    val_split: float = 0.1
    test_split: float = 0.1
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class TrainingConfig(BaseModel):
    """Training loop hyperparameters and checkpointing settings.

    epochs: total training epochs.
    batch_size: total batch size (P*K); P and K are derived from PKSampler settings.
    lr: initial learning rate for AdamW.
    scheduler: LR schedule — "cosine" (recommended), "linear", or "constant".
    warmup_steps: steps to linearly ramp LR from ~0 to lr.
                  Critical for transformer fine-tuning — avoids large initial gradients.
    grad_clip: max gradient norm for clip_grad_norm_ (1.0 is standard for BERT-family).
    mixed_precision: float16 AMP on CUDA for ~2x speed on Ampere+ GPUs.
    output_dir: root for all run outputs; each run writes to output_dir/<run_name>/.
    checkpoint_every_n: save a numbered checkpoint every N epochs.
    keep_last_n: delete numbered checkpoints older than the last N (0 = keep all).
    save_best: maintain checkpoint_best.pt for the lowest val/loss epoch.
    """
    model_config = ConfigDict(extra="forbid")

    epochs: int = 10
    batch_size: int = 64
    lr: float = 2e-5
    scheduler: str = "cosine"
    warmup_steps: int = 100
    grad_clip: float = 1.0
    mixed_precision: bool = True
    output_dir: str = "runs"          # root directory for all experiment outputs
    checkpoint_every_n: int = 1       # save a checkpoint every N epochs
    keep_last_n: int = 3              # keep only the N most recent epoch checkpoints (0 = keep all)
    save_best: bool = True            # maintain a checkpoint_best.pt tracking lowest val/loss


class WandbConfig(BaseModel):
    """Weights & Biases experiment tracking settings."""
    model_config = ConfigDict(extra="forbid")

    project: str = "email-fraud-detection"
    entity: str | None = None   # wandb team / username; None = personal workspace
    name: str | None = None     # human-readable run name shown in W&B dashboard
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class RunpodConfig(BaseModel):
    """Optional RunPod GPU pod settings for remote training.

    Only used when scripts/train.py is invoked with --runpod.
    """
    model_config = ConfigDict(extra="forbid")

    gpu_type: str = "NVIDIA A100-SXM4-80GB"
    disk_gb: int = 50
    container_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Root config composed from base.yaml + experiment override file.

    confidence_tiers maps a textual k-range (e.g. "1-4") to a tier label
    (e.g. "low") used by heads and the profile store to qualify predictions.
    """

    model_config = ConfigDict(extra="forbid")

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    runpod: RunpodConfig | None = None
    confidence_tiers: dict[str, str] = Field(
        default_factory=lambda: {
            "1-4": "low",
            "5-9": "medium",
            "10-24": "high",
            "25+": "very_high",
        }
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_BASE_YAML = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
_PROJECT_ROOT = _BASE_YAML.parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*; override keys win.

    Nested dicts are merged recursively rather than replaced wholesale.
    This lets experiment YAMLs override a single nested key (e.g. training.lr)
    without repeating all sibling keys from the base config.

    Example:
        base     = {"training": {"lr": 1e-4, "epochs": 10}}
        override = {"training": {"lr": 5e-5}}
        result   = {"training": {"lr": 5e-5, "epochs": 10}}
    """
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            # Non-dict values (scalars, lists) are replaced wholesale.
            merged[key] = val
    return merged


def load_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment config.

    Strategy:
    1. Load configs/base.yaml for defaults.
    2. Load the experiment file at *path*.
    3. Deep-merge (experiment keys win).
    4. Parse the merged dict through ExperimentConfig (Pydantic v2).
    """
    base_path = _BASE_YAML
    base_data: dict[str, Any] = {}
    if base_path.exists():
        with open(base_path) as fh:
            base_data = yaml.safe_load(fh) or {}

    with open(path) as fh:
        experiment_data: dict[str, Any] = yaml.safe_load(fh) or {}

    merged = _deep_merge(base_data, experiment_data)
    cfg = ExperimentConfig.model_validate(merged)

    # Resolve relative data paths against the project root so train/evaluate work
    # from any CWD (e.g. on RunPod where the shell may be in /workspace, not the repo).
    def _abs(p: str) -> str:
        pp = Path(p)
        return str(_PROJECT_ROOT / pp if not pp.is_absolute() else pp)

    cfg.data.data_dir = _abs(cfg.data.data_dir)
    cfg.data.processed_dir = _abs(cfg.data.processed_dir)
    return cfg

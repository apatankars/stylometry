"""Trainer — contrastive training loop with checkpointing and wandb logging.

Checkpoint layout under <output_dir>/::

    checkpoint_epoch_001.pt
    checkpoint_epoch_002.pt
    ...
    checkpoint_last.pt      <- always overwritten with the latest epoch
    checkpoint_best.pt      <- lowest val/loss seen so far (if save_best=True)
    config.yaml             <- copy of the experiment config (written by train.py)

Each .pt file contains::

    {
        "epoch":               int,
        "model_state_dict":    ...,
        "optimizer_state_dict",...,
        "scheduler_state_dict",...,
        "scaler_state_dict":   ... | None,
        "best_val_loss":       float,
    }

Resume::

    Pass resume_from=<path-to-.pt> to __init__; training restarts from
    epoch+1 with all optimizer/scheduler/scaler states restored.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from email_fraud.config import TrainingConfig, WandbConfig
from email_fraud.encoders.base import BaseEncoder
from email_fraud.heads.base import BaseHead
from email_fraud.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Contrastive training loop with checkpointing, resume, and wandb logging.

    Args:
        model:        BaseEncoder to train.
        loss_fn:      BaseLoss instance.
        head:         BaseHead for sender profiles (accumulated but not yet scored).
        config:       TrainingConfig.
        wandb_config: WandbConfig.
        output_dir:   Directory to write checkpoints and logs.  Created if absent.
        resume_from:  Path to a checkpoint .pt file to resume from.
        device:       Torch device string; auto-detected if None.
    """

    def __init__(
        self,
        model: BaseEncoder,
        loss_fn: BaseLoss,
        head: BaseHead,
        config: TrainingConfig,
        wandb_config: WandbConfig,
        output_dir: Path | str,
        resume_from: Path | str | None = None,
        device: str | None = None,
        eval_config_path: str | Path | None = None,
        eval_data_dir: str | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.head = head
        self.config = config
        self.wandb_config = wandb_config
        self.output_dir = Path(output_dir)
        # Optional: path to experiment config & processed eval data dir
        self.eval_config_path = Path(eval_config_path) if eval_config_path is not None else None
        self.eval_data_dir = Path(eval_data_dir) if eval_data_dir is not None else None
        # Auto-detect GPU if available; fall back to CPU for local development.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        # Filter to trainable params only — frozen backbone params must not be
        # passed to the optimizer (they have requires_grad=False and zero grad).
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found in the encoder. "
                "Set freeze_backbone=False, add a LoRA config, or set projection_dim."
            )
        # AdamW decouples weight decay from gradient update, which matters for
        # transformers where L2 regularization on the Adam adaptive scale is wrong.
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)

        # GradScaler for automatic mixed precision (AMP): keeps a loss scale factor
        # that prevents float16 underflow.  Only used on CUDA; CPU doesn't support AMP.
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler()
            if config.mixed_precision and self.device != "cpu"
            else None
        )

        self._start_epoch: int = 1
        self._best_val_loss: float = float("inf")

        # Restore model, optimizer, scaler, and epoch counter from checkpoint.
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Run the full training loop from _start_epoch to config.epochs."""
        import wandb

        # wandb.init with resume="allow" means: if a run id already exists in
        # the output_dir (from a previous run), resume that run's metrics rather
        # than starting a new one.  This keeps training curves continuous.
        run = wandb.init(
            project=self.wandb_config.project,
            entity=self.wandb_config.entity,
            name=self.wandb_config.name,
            tags=self.wandb_config.tags,
            notes=self.wandb_config.notes,
            dir=str(self.output_dir),
            config={
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "scheduler": self.config.scheduler,
                "warmup_steps": self.config.warmup_steps,
                "mixed_precision": self.config.mixed_precision,
                "output_dir": str(self.output_dir),
            },
            resume="allow",
        )

        # Build the scheduler after init so we know steps_per_epoch.
        scheduler = self._build_scheduler(len(train_loader))

        try:
            for epoch in range(self._start_epoch, self.config.epochs + 1):
                train_loss = self._train_epoch(train_loader, scheduler)
                val_metrics = self._validate(val_loader)
                val_loss = val_metrics.get("val/loss", float("inf"))
                current_lr = self.optimizer.param_groups[0]["lr"]

                wandb.log({"epoch": epoch, "train/loss": train_loss, "train/lr": current_lr, **val_metrics})
                logger.info(
                    "Epoch %d/%d  train/loss=%.4f  %s",
                    epoch,
                    self.config.epochs,
                    train_loss,
                    "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                )

                # Periodic epoch checkpoint (e.g. every 5 epochs for long runs).
                if epoch % self.config.checkpoint_every_n == 0:
                    self._save_epoch_checkpoint(epoch, val_loss)

                # checkpoint_last.pt is always written — used by the eval script
                # and as a safe resume point even between periodic checkpoints.
                self._save_last_checkpoint(epoch, val_loss)

                # checkpoint_best.pt tracks the epoch with the lowest val loss.
                # This is the checkpoint to use for profile-building / inference,
                # not necessarily the final epoch.
                if self.config.save_best and val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._save_best_checkpoint(epoch, val_loss)

                # Prune old epoch checkpoints to keep disk usage bounded.
                if self.config.keep_last_n > 0:
                    self._prune_old_checkpoints(epoch)

                # Periodic PAN evaluation (external script) every 5 epochs if configured.
                # This runs evaluate.py as a subprocess so it uses a fresh Python
                # process (avoids GPU memory accumulation during long training runs).
                if epoch % 5 == 0 and self.eval_config_path is not None and self.eval_data_dir is not None:
                    import subprocess
                    import sys

                    # project root is four parents up from this file (src/email_fraud/training)
                    project_root = Path(__file__).resolve().parents[4]
                    eval_script = project_root / "scripts" / "evaluate.py"
                    checkpoint_path = self.output_dir / "checkpoint_last.pt"
                    cmd = [
                        sys.executable,
                        str(eval_script),
                        "--config",
                        str(self.eval_config_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--data-dir",
                        str(self.eval_data_dir),
                    ]
                    logger.info("Running periodic evaluation (every 5 epochs): %s", " ".join(cmd))
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        logger.warning("Periodic evaluation failed (exit %s): %s", e.returncode, e)
                    except Exception as e:
                        logger.warning("Periodic evaluation failed: %s", e)

        finally:
            # Always finish wandb run even if training is interrupted, so the
            # run appears as "finished" not "crashed" in the wandb dashboard.
            wandb.finish()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _checkpoint_payload(self, epoch: int, scheduler: Any, val_loss: float) -> dict:
        return {
            "epoch": epoch,
            "best_val_loss": self._best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler_state,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }

    def _save_epoch_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(self._build_payload(epoch, val_loss), path)
        logger.debug("Saved epoch checkpoint: %s", path)

    def _save_last_checkpoint(self, epoch: int, val_loss: float) -> None:
        # Overwritten every epoch — always reflects the most recent state.
        path = self.output_dir / "checkpoint_last.pt"
        torch.save(self._build_payload(epoch, val_loss), path)

    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.output_dir / "checkpoint_best.pt"
        torch.save(self._build_payload(epoch, val_loss), path)
        logger.info("New best val/loss=%.4f at epoch %d → %s", val_loss, epoch, path)

    def _build_payload(self, epoch: int, val_loss: float) -> dict:
        """Assemble the dict that gets torch.save()'d to disk."""
        return {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self._best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # _scheduler_state is updated after every step in _train_epoch
            # so it's always current when _build_payload is called.
            "scheduler_state_dict": self._scheduler_state,
            # None when running on CPU (scaler is disabled)
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info("Resuming from checkpoint: %s", path)
        # map_location ensures we can load a GPU checkpoint onto CPU
        # (and vice versa) without error.
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scaler_state_dict") and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        self._best_val_loss = payload.get("best_val_loss", float("inf"))
        # Resume from the epoch AFTER the saved one.
        self._start_epoch = payload["epoch"] + 1
        # Stash scheduler state here; it can only be loaded after the scheduler
        # is built in train() (since we don't know steps_per_epoch yet).
        self._resume_scheduler_state = payload.get("scheduler_state_dict")
        logger.info("Resuming from epoch %d (best val/loss so far: %.4f)",
                    payload["epoch"], self._best_val_loss)

    def _prune_old_checkpoints(self, current_epoch: int) -> None:
        """Delete epoch checkpoints older than the last keep_last_n."""
        n = self.config.keep_last_n
        # Sort by epoch number (extracted from the filename suffix).
        epoch_ckpts = sorted(
            self.output_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        # Keep only the last n; delete the rest.
        for old in epoch_ckpts[:-n]:
            old.unlink()
            logger.debug("Pruned old checkpoint: %s", old)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    # Scheduler state is built inside train() and we need it for checkpointing.
    # We store it as an instance attr after each step so _build_payload can read it.
    _scheduler_state: dict | None = None
    _resume_scheduler_state: dict | None = None

    def _train_epoch(self, loader: DataLoader, scheduler: Any) -> float:
        """Single training epoch; returns mean loss over all batches."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="train", leave=False):
            # batch is an EpisodeBatch (from episode_collate); texts are raw strings,
            # labels are integer sender ids (assigned by first-appearance order).
            texts: list[str] = batch.texts
            labels: torch.Tensor = batch.labels.to(self.device)

            # Tokenize on CPU (tokenizer is CPU-bound), then move tensors to device.
            token_dict = self.model.tokenize(texts)
            token_dict = {k: v.to(self.device) for k, v in token_dict.items()}

            self.optimizer.zero_grad()

            if self.scaler is not None:
                # AMP: run forward pass in float16 for speed, keeping a float32
                # master copy of weights.  The scaler adjusts the loss scale to
                # prevent float16 underflow during backward.
                with torch.amp.autocast(device_type=self.device):
                    embeddings = self.model.encode(**token_dict)
                    loss = self.loss_fn(embeddings, labels)
                self.scaler.scale(loss).backward()
                # Unscale before clip_grad_norm so the norm is in "true" units,
                # not inflated by the loss scale factor.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard fp32 path (CPU or if mixed_precision=False).
                embeddings = self.model.encode(**token_dict)
                loss = self.loss_fn(embeddings, labels)
                loss.backward()
                # Gradient clipping prevents exploding gradients in transformer fine-tuning;
                # grad_clip=1.0 is a safe default from the original BERT paper.
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            # Scheduler steps per batch (not per epoch) for smooth LR curves.
            scheduler.step()
            # Cache scheduler state so _build_payload can read it without
            # passing the scheduler object around.
            self._scheduler_state = scheduler.state_dict()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Compute val loss and embedding-space classification metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_embs: list[torch.Tensor] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="val", leave=False):
                texts: list[str] = batch.texts
                labels: torch.Tensor = batch.labels.to(self.device)

                token_dict = self.model.tokenize(texts)
                token_dict = {k: v.to(self.device) for k, v in token_dict.items()}

                embeddings = self.model.encode(**token_dict)
                loss = self.loss_fn(embeddings, labels)
                total_loss += loss.item()
                n_batches += 1

                all_embs.append(embeddings.detach().cpu())
                all_labels.extend(labels.cpu().tolist())

        metrics: dict[str, float] = {"val/loss": total_loss / max(n_batches, 1)}
        if len(all_embs) > 1:
            embs = torch.cat(all_embs, dim=0)
            labels_t = torch.tensor(all_labels)
            metrics.update(self._compute_embedding_metrics(embs, labels_t))
        return metrics

    def _compute_embedding_metrics(
        self, embs: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """1-NN accuracy, macro F1, and intra/inter cosine similarities."""
        import torch.nn.functional as F
        from sklearn.metrics import f1_score

        N = embs.size(0)
        if N < 2:
            return {}

        embs_norm = F.normalize(embs, dim=1)          # (N, d) unit vectors
        sim = embs_norm @ embs_norm.T                 # (N, N) cosine similarities

        # Leave-one-out 1-NN: exclude self by masking diagonal
        sim_loo = sim.clone()
        sim_loo.fill_diagonal_(-2.0)
        nn_labels = labels[sim_loo.argmax(dim=1)]

        knn_acc = (nn_labels == labels).float().mean().item()
        macro_f1 = float(
            f1_score(labels.numpy(), nn_labels.numpy(), average="macro", zero_division=0)
        )

        # Intra-class similarity (same sender, excluding self) and inter-class
        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        eye = torch.eye(N, dtype=torch.bool)
        intra_mask = same & ~eye
        inter_mask = ~same

        intra_sim = sim[intra_mask].mean().item() if intra_mask.any() else 0.0
        inter_sim = sim[inter_mask].mean().item() if inter_mask.any() else 0.0

        return {
            "val/knn_acc": knn_acc,
            "val/macro_f1": macro_f1,
            "val/intra_cos_sim": intra_sim,
            "val/inter_cos_sim": inter_sim,
        }

    def _build_scheduler(self, steps_per_epoch: int) -> Any:
        """Construct the LR scheduler based on config.scheduler.

        All schedulers step per *batch* (not per epoch), so total_steps is
        steps_per_epoch * epochs.  Warmup linearly ramps LR from ~0 to the
        initial lr over the first warmup_steps steps — important for transformers
        to avoid large gradient updates on the cold, randomly-initialized head.
        """
        total_steps = steps_per_epoch * self.config.epochs
        warmup = self.config.warmup_steps

        if self.config.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            # Warmup phase: linear ramp from near-zero to initial lr.
            warmup_sched = LinearLR(
                self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup
            )
            # Main phase: cosine annealing from initial lr to ~0 over remaining steps.
            cosine_sched = CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps - warmup, 1)
            )
            # SequentialLR chains the two: run warmup_sched for `warmup` steps,
            # then hand off to cosine_sched.
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup],
            )
        elif self.config.scheduler == "linear":
            from torch.optim.lr_scheduler import LinearLR

            # Linear decay from initial lr to 0 over total_steps.
            scheduler = LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
            )
        elif self.config.scheduler == "constant":
            from torch.optim.lr_scheduler import ConstantLR

            # No decay — lr stays fixed.  Useful for debugging or short runs.
            scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=total_steps)
        else:
            raise ValueError(
                f"Unknown scheduler '{self.config.scheduler}'. "
                "Choose from: 'cosine', 'linear', 'constant'."
            )

        # If resuming, restore the scheduler's step counter so the LR curve
        # continues from where it left off (not restart from step 0).
        if self._resume_scheduler_state is not None:
            scheduler.load_state_dict(self._resume_scheduler_state)
            self._resume_scheduler_state = None  # consumed; don't restore twice

        self._scheduler_state = scheduler.state_dict()
        return scheduler

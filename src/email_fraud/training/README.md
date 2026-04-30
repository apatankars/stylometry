# training/

The contrastive training loop: fits the encoder on labeled email batches using a contrastive loss, with checkpointing, resume, mixed-precision, and W&B logging.

---

## Files

### `trainer.py` — `Trainer`

The single training class. Wires together the encoder, loss function, head, and data loaders for the full training loop.

---

## Training loop overview

```
Trainer.train(train_loader, val_loader)
│
├─ wandb.init()          ← initialize experiment tracking
├─ _build_scheduler()    ← build LR scheduler (cosine/linear/constant + warmup)
│
└─ for epoch in range(start, epochs+1):
       │
       ├─ _train_epoch(train_loader, scheduler)
       │     └─ for each EpisodeBatch:
       │           tokenize → encode → loss → backward → clip_grad → step → scheduler.step
       │
       ├─ _validate(val_loader)
       │     └─ eval mode, no_grad, compute val/loss
       │
       ├─ wandb.log(epoch, train/loss, val/loss)
       │
       ├─ _save_epoch_checkpoint()   ← every checkpoint_every_n epochs
       ├─ _save_last_checkpoint()    ← every epoch (resume point)
       ├─ _save_best_checkpoint()    ← when val/loss improves
       ├─ _prune_old_checkpoints()   ← keep only last keep_last_n
       │
       └─ periodic evaluate.py      ← every 5 epochs if eval paths configured
```

---

## Constructor

```python
Trainer(
    model,         # BaseEncoder — the model to train
    loss_fn,       # BaseLoss — contrastive objective
    head,          # BaseHead — profiles accumulated during training (not yet scored)
    config,        # TrainingConfig
    wandb_config,  # WandbConfig
    output_dir,    # where to write checkpoints
    resume_from,   # optional path to a .pt checkpoint to resume from
    device,        # "cuda" / "cpu" — auto-detected if None
    eval_config_path,   # optional — path to experiment YAML for periodic eval
    eval_data_dir,      # optional — path to processed eval data
)
```

**Important**: raises `ValueError` at construction if the model has no trainable parameters (all frozen, no LoRA, no projection).

---

## Mixed-precision (AMP)

When `config.mixed_precision=True` and running on CUDA, `torch.amp.GradScaler` is used:

1. Forward pass runs in `float16` (faster on Ampere+ GPUs, ~2× speedup)
2. `scaler.scale(loss).backward()` — gradients are scaled to prevent float16 underflow
3. `scaler.unscale_()` — unscale before `clip_grad_norm_` so the norm is in true units
4. `scaler.step(optimizer)` — skips update if gradients contain inf/nan (from overflow)
5. `scaler.update()` — adjusts scale factor for next step

Disabled on CPU automatically.

---

## Checkpoints

Each `.pt` file contains:

```python
{
    "epoch":                int,
    "val_loss":             float,
    "best_val_loss":        float,
    "model_state_dict":     dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict | None,
    "scaler_state_dict":    dict | None,
}
```

| File | When written | Purpose |
|------|-------------|---------|
| `checkpoint_epoch_NNN.pt` | Every `checkpoint_every_n` epochs | Long-term history |
| `checkpoint_last.pt` | Every epoch (overwritten) | Safe resume point |
| `checkpoint_best.pt` | When val/loss improves | Best model for inference |

To resume: pass `resume_from="path/to/checkpoint_last.pt"` to `Trainer.__init__`.

---

## Learning rate scheduling

All schedulers step **per batch** (not per epoch) for smooth LR curves.

| Scheduler | Behavior |
|-----------|---------|
| `"cosine"` | Linear warmup for `warmup_steps` steps, then cosine decay to ~0 |
| `"linear"` | Linear decay from initial lr to 0 over all steps |
| `"constant"` | Fixed lr throughout |

`"cosine"` is recommended. Warmup prevents large gradient updates before the projection head is initialized.

---

## Periodic evaluation

Every 5 epochs, if `eval_config_path` and `eval_data_dir` are set, the trainer runs `scripts/evaluate.py` as a subprocess using `checkpoint_last.pt`. This:
- Computes PAN metrics (AUC, EER, c@1, F0.5u) on the held-out test set
- Runs in a fresh process to avoid accumulating GPU memory
- Failures are logged as warnings (don't abort training)

---

## Configuration reference

```yaml
training:
  epochs: 10
  batch_size: 64
  lr: 2e-5
  scheduler: cosine        # cosine | linear | constant
  warmup_steps: 100
  grad_clip: 1.0
  mixed_precision: true
  output_dir: runs
  checkpoint_every_n: 1
  keep_last_n: 3
  save_best: true
```

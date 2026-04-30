# utils/

Shared utilities: logging configuration and RunPod cloud GPU management.

---

## Files

### `logging.py`

Logging helpers. Call `setup_logging()` once at the top of each script; all other modules use the standard `logging.getLogger(__name__)` pattern.

| Function | Description |
|----------|-------------|
| `setup_logging(level="INFO")` | Configure root logger to stdout with format `timestamp  LEVEL  module  message` |
| `log_config(config, logger)` | Pretty-print any Pydantic `ExperimentConfig` as JSON to the logger |
| `wandb_watch(model, log_freq=100)` | Call `wandb.watch(model)` if a wandb run is active; no-op otherwise |

`wandb_watch` is a no-op when wandb is not installed or not initialized. This lets you import it unconditionally without worrying about whether wandb is configured.

---

### `runpod.py`

RunPod GPU pod launch and teardown. Used by `scripts/train.py --runpod` to spin up a remote A100 pod for long training runs.

**Requires**: `RUNPOD_API_KEY` environment variable.

| Function | Description |
|----------|-------------|
| `launch_pod(gpu_type, disk_gb, container_image)` | Create a pod and return its `pod_id` |
| `wait_for_running(pod_id, poll_interval=10, timeout=300)` | Block until pod reaches `RUNNING` state |
| `terminate_pod(pod_id)` | Gracefully terminate pod; safe to call even if already stopped |

`terminate_pod` catches exceptions and logs a warning rather than raising — this is intentional so training script teardown doesn't fail if the pod was already terminated (e.g. due to timeout or manual intervention).

#### Typical usage in train.py

```python
pod_id = launch_pod(config.runpod.gpu_type, config.runpod.disk_gb, config.runpod.container_image)
try:
    wait_for_running(pod_id)
    # ... run training remotely ...
finally:
    terminate_pod(pod_id)   # always clean up, even on error
```

#### RunpodConfig reference

```yaml
runpod:
  gpu_type: "NVIDIA A100-SXM4-80GB"
  disk_gb: 50
  container_image: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
```

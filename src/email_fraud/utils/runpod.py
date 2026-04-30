"""RunPod launch and teardown helpers.

These utilities let scripts/train.py optionally spin up a remote GPU pod for
long training runs and tear it down when training completes.

Requires: RUNPOD_API_KEY in environment (or .env file).
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


def _client():
    """Return an initialised runpod client, raising clearly if not configured."""
    try:
        import runpod  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "runpod package not installed. "
            "Add it to pyproject.toml or: pip install runpod"
        ) from exc

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "RUNPOD_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    runpod.api_key = api_key
    return runpod


def launch_pod(
    gpu_type: str,
    disk_gb: int,
    container_image: str,
    name: str = "email-fraud-train",
    volume_mount_path: str = "/workspace",
) -> str:
    """Create a RunPod GPU pod and return its pod_id.

    Args:
        gpu_type:          RunPod GPU type string (e.g. "NVIDIA A100-SXM4-80GB").
        disk_gb:           Container + volume disk size.
        container_image:   Docker image to use.
        name:              Human-readable pod name.
        volume_mount_path: Where to mount the persistent volume.

    Returns:
        pod_id string.
    """
    rp = _client()
    logger.info("Launching RunPod pod: gpu=%s image=%s", gpu_type, container_image)

    pod = rp.create_pod(
        name=name,
        image_name=container_image,
        gpu_type_id=gpu_type,
        cloud_type="SECURE",
        container_disk_in_gb=disk_gb,
        volume_in_gb=disk_gb,
        volume_mount_path=volume_mount_path,
        ports="22/tcp",
    )
    pod_id: str = pod["id"]
    logger.info("Pod created: id=%s", pod_id)
    return pod_id


def wait_for_running(pod_id: str, poll_interval: int = 10, timeout: int = 300) -> None:
    """Block until the pod reaches RUNNING state or timeout is exceeded."""
    rp = _client()
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = rp.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")
        logger.debug("Pod %s status: %s", pod_id, status)
        if status == "RUNNING":
            return
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Pod {pod_id} did not reach RUNNING within {timeout}s."
    )


def terminate_pod(pod_id: str) -> None:
    """Terminate a pod by id.  Safe to call even if already terminated."""
    rp = _client()
    try:
        rp.terminate_pod(pod_id)
        logger.info("Terminated pod %s", pod_id)
    except Exception as exc:
        logger.warning("Failed to terminate pod %s: %s", pod_id, exc)

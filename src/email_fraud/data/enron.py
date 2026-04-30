"""EnronDataset — loads preprocessed Enron splits from Arrow format.

Usage:
    Run scripts/prepare_data.py first to build the Arrow dataset, then this
    class loads it.  The dataset is registered as "enron" in the component
    registry and can be resolved via resolve("dataset", "enron").

Split semantics:
    "train"      — senders used for contrastive training.
    "validation" — held-out senders for hyperparameter search / early stopping.
    "test"       — final held-out senders; never used during model selection.

    Splits are sender-disjoint: no sender appears in more than one split.
    This is critical for authorship attribution — a model that has seen a
    sender's emails at train time has an unfair advantage at test time.
    A sender-disjoint split simulates the real deployment scenario where
    the model must recognise *new* senders it has never seen during training.
"""

from __future__ import annotations

from email_fraud.data.base import BaseDataset
from email_fraud.config import DataConfig
from email_fraud.registry import register


@register("dataset", "enron")
class EnronDataset(BaseDataset):
    """HuggingFace Arrow-backed Enron email dataset.

    Arrow format (via the `datasets` library) stores the data in a columnar
    memory-mapped format, which means:
      - Very fast random access by index (no sequential scan needed).
      - Low memory footprint — data stays on disk until accessed.
      - Consistent across Python versions (no pickle issues).

    The dataset is created by scripts/prepare_data.py which:
      1. Reads raw Enron .msg files from data_dir.
      2. Preprocesses each email via clean_email_raw().
      3. Filters senders with fewer than min_emails_per_sender emails.
      4. Assigns sender-disjoint train/validation/test splits.
      5. Saves the result as a DatasetDict in Arrow format to processed_dir.

    Args:
        config: DataConfig containing processed_dir path and split name.
        split:  One of "train", "validation", "test".
    """

    def __init__(self, config: DataConfig, split: str = "train") -> None:
        # Lazy import: `datasets` is only needed when actually loading data,
        # not at import time (important for scripts that import the package
        # without loading data).
        from datasets import load_from_disk

        dataset_dict = load_from_disk(config.processed_dir)
        if split not in dataset_dict:
            available = list(dataset_dict.keys())
            raise ValueError(
                f"Split '{split}' not found in {config.processed_dir}. "
                f"Available: {available}. Run scripts/prepare_data.py first."
            )
        ds = dataset_dict[split]
        # Load both columns into plain Python lists for O(1) random access.
        # The HuggingFace Dataset object supports __getitem__ but is slower
        # for random access than a pre-loaded list.
        self._texts: list[str] = ds["text"]
        self._sender_ids_list: list[str] = ds["sender_id"]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> tuple[str, str]:
        # Returns (email_text, sender_id) — the format expected by episode_collate.
        return self._texts[index], self._sender_ids_list[index]

    @property
    def sender_ids(self) -> list[str]:
        # PKSampler calls this once at construction to build per-sender index maps.
        return self._sender_ids_list

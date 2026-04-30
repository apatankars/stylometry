"""BaseEncoder ABC.

All encoders must produce L2-normalized embeddings of shape (B, d_model) and
expose a tokenize() method so the rest of the pipeline never needs to know
which tokenizer is in use.  The MODEL_TYPE slot allows downstream code (e.g.
the LUAR episode pooling path) to branch on architecture family without
isinstance checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base for all text encoders used in the contrastive pipeline.

    Subclasses must:
    - Return L2-normalized embeddings from encode().
    - Expose the output dimensionality via the embedding_dim property.
    - Provide a tokenize() method that returns a HuggingFace-style dict of
      tensors so callers never need to import a specific tokenizer.

    MODEL_TYPE is a class-level string tag (e.g. "subword", "char", "luar")
    used for architecture-specific branching without isinstance checks.
    """

    # Class-level tag for architecture family branching.
    # HFEncoder sets this to "subword"; a character-level model might use "char".
    # The training loop and pooling logic read this to decide code paths
    # (e.g. whether to reshape input for episode-level pooling).
    MODEL_TYPE: ClassVar[str] = "base"

    @abstractmethod
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Encode a batch of tokenized inputs.

        Args:
            input_ids: (B, L) or (B, K, L) for episode-style models.
            attention_mask: same shape as input_ids.
            **kwargs: model-specific extras (e.g. token_type_ids).

        Returns:
            Float tensor of shape (B, d_model), L2-normalized.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output dimensionality of the encoder (post-pooling).

        Used by the head and loss functions to validate that the embedding
        dimension matches what they were configured with.
        """
        ...

    @abstractmethod
    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize *texts* and return a HuggingFace-style dict of tensors.

        The returned dict must contain at least ``input_ids`` and
        ``attention_mask`` and be ready to unpack directly into encode().

        Centralising tokenization here means training, validation, and
        inference all use the exact same tokenizer settings (padding, truncation,
        max_length), preventing train/serve skew.
        """
        ...

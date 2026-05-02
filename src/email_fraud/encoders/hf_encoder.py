"""HuggingFace encoder — wraps any AutoModel behind the BaseEncoder interface.

Supports RoBERTa, LUAR, ModernBERT, CANINE, etc. because all load via
AutoModel.from_pretrained.  LoRA is applied via peft when config.lora is set.

Pooling strategies
------------------
mean          : attention-masked mean over token dimension (default)
cls           : first [CLS] token embedding
luar_episode  : episode-level mean-pool; expects input shaped (B, K, L) and
                returns (B, d_model) by averaging over the K-episode dim.
                See Rivera et al. "Learning Universal Authorship Representations"
                (EMNLP 2021, arXiv:2107.10882).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from email_fraud.config import EncoderConfig
from email_fraud.encoders.base import BaseEncoder

# @register("encoder", "hf") — Rivera et al. LUAR (arXiv:2107.10882) and
# any AutoModel-compatible checkpoint (RoBERTa, ModernBERT, CANINE, etc.)
from email_fraud.registry import register


class _LUARPeftAdapter(nn.Module):
    """Wraps LUAR before LoRA is applied so PEFT's injected inputs_embeds kwarg is absorbed.

    PEFT's PeftModelForFeatureExtraction.forward always passes inputs_embeds=None
    to the base model, but LUAR.forward() only accepts (input_ids, attention_mask,
    output_attentions, document_batch_size). This adapter sits between PEFT and LUAR,
    absorbing the extra kwarg. LoRA target modules are still found because PEFT
    scans named_modules() which reaches through _luar.
    """

    def __init__(self, luar_model: nn.Module) -> None:
        super().__init__()
        self._luar = luar_model
        self.config = luar_model.config

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self._luar(input_ids=input_ids, attention_mask=attention_mask)


@register("encoder", "hf")
class HFEncoder(BaseEncoder):
    """HuggingFace AutoModel encoder with optional LoRA fine-tuning.

    Accepts an EncoderConfig; if config.lora is set, wraps the backbone with
    peft.get_peft_model using LoraConfig.  Supports mean, cls, and
    luar_episode pooling strategies.
    """

    # "subword" signals to downstream code (e.g. the LUAR episode path) that
    # this encoder uses a byte-pair / WordPiece tokenizer.  A character-level
    # model (CANINE) would still use this class but could override the tag.
    MODEL_TYPE = "subword"

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        # Lazy import keeps transformers out of the import graph until actually needed.
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        # Tokenizer is stored on the encoder so tokenize() and encode() are
        # always paired — prevents train/serve skew from using different tokenizers.
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=True
        )

        backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        # Transformers 5.x promotes non-persistent buffers from meta→device via
        # torch.empty_like (uninitialized). Re-zero any token_type_ids buffers,
        # which must be all-zeros for correct RoBERTa-family inference.
        for module in backbone.modules():
            if hasattr(module, "token_type_ids") and isinstance(module.token_type_ids, torch.Tensor):
                module.token_type_ids.zero_()

        if config.lora is not None:
            # LoRA inserts low-rank adapter matrices into the attention layers.
            # Only the adapter weights (r × d_model per targeted module) are
            # trainable; the original backbone weights are frozen.  This lets
            # us fine-tune a 125M-param RoBERTa with ~0.3% of the parameters.
            from peft import LoraConfig, TaskType, get_peft_model

            # LUAR's forward() doesn't accept inputs_embeds, which PEFT's
            # PeftModelForFeatureExtraction injects unconditionally. Wrap before
            # get_peft_model so the adapter (not LUAR) absorbs that kwarg.
            if config.pooling == "luar_episode":
                backbone = _LUARPeftAdapter(backbone)

            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora.r,           # rank of the adapter matrices
                lora_alpha=config.lora.alpha,  # scaling factor (alpha/r applied at forward)
                target_modules=config.lora.target_modules,  # typically ["query", "value"]
                lora_dropout=config.lora.dropout,
                bias="none",  # don't add bias to adapter layers
            )
            backbone = get_peft_model(backbone, lora_cfg)

        if config.freeze_backbone and config.lora is None:
            # Hard-freeze without LoRA: backbone is a fixed feature extractor.
            # Only the projection head (if any) and contrastive loss are trained.
            # Useful for quick experiments or when compute is tight.
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        # LUAR uses embedding_size; standard transformers use hidden_size.
        backbone_dim: int = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "embedding_size")

        # Optional trainable projection head (useful for frozen-backbone experiments)
        # Projects from backbone_dim → projection_dim before L2 normalization.
        # A smaller projection_dim (e.g. 128) makes the loss landscape easier to
        # optimize and reduces memory when computing full pairwise distance matrices.
        if config.projection_dim is not None:
            self.projection: nn.Linear | None = nn.Linear(backbone_dim, config.projection_dim, bias=False)
            self._embedding_dim = config.projection_dim
        else:
            self.projection = None
            self._embedding_dim = backbone_dim

    # ------------------------------------------------------------------
    # BaseEncoder interface
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def episode_k(self) -> int | None:
        """Number of emails per LUAR episode; None for non-episode-pooling encoders."""
        if self.config.pooling == "luar_episode":
            return self.config.episode_k
        return None

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        # padding=True pads all sequences in the batch to the same length.
        # truncation=True + max_length silently clips anything longer than
        # config.max_length tokens — critical for email bodies which vary wildly.
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass through backbone + pooling + L2 normalization.

        Args:
            input_ids: (B, L) for mean/cls pooling; (B, K, L) for luar_episode.
            attention_mask: same shape as input_ids.

        Returns:
            (B, d_model) L2-normalized float tensor.
        """
        # TODO: implement full forward pass; stub returns zero embeddings for
        # shape correctness so the pipeline can be tested without weights.
        pooling = self.config.pooling

        # LUAR episode pooling requires a different input shape (B, K, L) and
        # its own forward path — branch before calling the backbone.
        if pooling == "luar_episode":
            if self.config.episode_k is None:
                raise ValueError(
                    "encoder.episode_k must be set when pooling='luar_episode'. "
                    "Set it to the number of emails per episode in your experiment config."
                )
            episode_k = self.config.episode_k
            n_total = input_ids.size(0)
            if n_total % episode_k != 0:
                raise ValueError(
                    f"Batch size {n_total} is not divisible by episode_k={episode_k}. "
                    "Ensure batch_size // emails_per_sender_k == P and "
                    "emails_per_sender_k // episode_k is a whole number."
                )
            n_episodes = n_total // episode_k
            # Reshape flat (P*K, L) → (P*(K/episode_k), episode_k, L).
            # PKSampler lays emails out as K contiguous emails per sender, so each
            # consecutive episode_k rows belong to the same sender.
            input_ids = input_ids.view(n_episodes, episode_k, -1)
            attention_mask = attention_mask.view(n_episodes, episode_k, -1)
            return self._encode_luar_episode(input_ids, attention_mask, **kwargs)

        # Standard (B, L) path for mean and cls pooling.
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden = outputs.last_hidden_state  # (B, L, d) — one vector per token

        if pooling == "cls":
            # The [CLS] token at position 0 is trained by some models (e.g. BERT,
            # RoBERTa) to aggregate sentence-level information, but it can be noisy
            # for longer, multi-topic texts like emails.
            pooled = last_hidden[:, 0, :]  # (B, d)
        elif pooling == "mean":
            # Attention-masked mean ignores padding tokens, giving a more
            # representative average over actual content tokens.
            pooled = self._mean_pool(last_hidden, attention_mask)
        else:
            raise ValueError(
                f"Unknown pooling strategy '{pooling}'. "
                "Choose from: 'mean', 'cls', 'luar_episode'."
            )

        # Optional projection squeezes d_backbone → d_projection.
        if self.projection is not None:
            pooled = self.projection(pooled)

        # L2 normalization so all embeddings lie on the unit hypersphere.
        # This makes cosine similarity equivalent to a dot product, which
        # simplifies the contrastive loss math and bounds distances to [0, 2].
        return F.normalize(pooled, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-masked mean pooling over the token dimension.

        Padding tokens have attention_mask=0 and are excluded from the mean,
        so the pooled vector is not diluted by padding.
        """
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1) — broadcast over d
        summed = (last_hidden * mask).sum(dim=1)     # (B, d) — zero-out padding tokens
        counts = mask.sum(dim=1).clamp(min=1e-9)     # (B, 1) — avoid division by zero
        return summed / counts

    def _encode_luar_episode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """LUAR-style episode pooling: (B, K, L) → (B, d).

        LUAR (Rivera et al. 2021) encodes an "episode" of K emails jointly.
        The full LUAR model expects (B, K, L) input and returns (B, embedding_dim)
        directly — its own attention + max-pool episode aggregation is used.
        """
        # Pass the full episode tensor to LUAR; it returns a plain (B, d) tensor.
        episode_embs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if self.projection is not None:
            episode_embs = self.projection(episode_embs)
        return F.normalize(episode_embs, p=2, dim=-1)

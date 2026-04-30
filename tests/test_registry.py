"""Smoke test: register a DummyEncoder, resolve it, verify identity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is on the path when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# DummyEncoder — minimal BaseEncoder subclass with no real weights
# ---------------------------------------------------------------------------

from email_fraud.encoders.base import BaseEncoder
from email_fraud.registry import REGISTRY, list_components, register, resolve


class DummyEncoder(BaseEncoder):
    """Minimal encoder for registry testing — never loads any model weights."""

    MODEL_TYPE = "dummy"

    def __init__(self) -> None:
        super().__init__()

    @property
    def embedding_dim(self) -> int:
        return 64

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        b = input_ids.shape[0]
        return torch.zeros(b, self.embedding_dim)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        n = len(texts)
        return {
            "input_ids": torch.zeros(n, 8, dtype=torch.long),
            "attention_mask": torch.ones(n, 8, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_resolve_returns_same_class():
    register("encoder", "dummy")(DummyEncoder)
    resolved = resolve("encoder", "dummy")
    assert resolved is DummyEncoder, (
        f"Expected DummyEncoder, got {resolved}"
    )


def test_resolved_class_is_instantiable():
    resolved = resolve("encoder", "dummy")
    instance = resolved()
    assert isinstance(instance, BaseEncoder)
    assert instance.embedding_dim == 64


def test_list_components_includes_dummy():
    components = list_components()
    assert "encoder" in components
    assert "dummy" in components["encoder"]


def test_resolve_unknown_kind_raises_key_error():
    with pytest.raises(KeyError, match="Unknown registry kind"):
        resolve("nonexistent_kind", "anything")


def test_resolve_unknown_name_raises_key_error_with_available():
    with pytest.raises(KeyError, match="Available"):
        resolve("encoder", "no_such_encoder_xyz")


def test_register_wrong_base_raises_type_error():
    class NotAnEncoder:
        pass

    with pytest.raises(TypeError, match="must be a subclass"):
        register("encoder", "bad")(NotAnEncoder)


def test_register_unknown_kind_raises_value_error():
    with pytest.raises(ValueError, match="Unknown registry kind"):
        register("unknown_kind", "foo")(DummyEncoder)


def test_double_register_same_class_is_idempotent():
    """Registering the same class under the same name twice should not error."""
    register("encoder", "dummy")(DummyEncoder)
    assert resolve("encoder", "dummy") is DummyEncoder


def test_encode_output_shape():
    enc = DummyEncoder()
    batch = enc.tokenize(["hello", "world"])
    out = enc.encode(**batch)
    assert out.shape == (2, 64)

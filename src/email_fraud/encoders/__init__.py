"""Encoder subpackage — import triggers @register side-effects."""

from email_fraud.encoders.base import BaseEncoder
from email_fraud.encoders.hf_encoder import HFEncoder

__all__ = ["BaseEncoder", "HFEncoder"]

"""Quick preview of LLM-generated synthetic emails for a handful of senders.

Loads a few real emails per sender, generates 2-3 synthetic imitations, and
prints both side-by-side so you can eyeball style transfer quality.  No dataset
is saved — this is purely for sanity-checking prompt quality and model output.

Usage:
    python scripts/preview_synthetic_emails.py \\
        --config configs/base.yaml \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --n-senders 2 \\
        --n-generate 2 \\
        --load-in-4bit

    # Pin a specific sender:
    python scripts/preview_synthetic_emails.py \\
        --config configs/base.yaml \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --sender "john.doe@enron.com" \\
        --load-in-4bit
"""

from __future__ import annotations

import argparse
import random
import textwrap
from collections import defaultdict

from datasets import load_from_disk

from email_fraud.config import load_config
from email_fraud.data.preprocessing import preprocess

# Re-use helpers from the generation script.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_emails import _build_prompt, _load_model, _generate_batch, _TOPICS

_DIVIDER = "─" * 72
_INDENT = "    "


def _wrap(text: str, width: int = 68) -> str:
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=_INDENT, subsequent_indent=_INDENT)
        if line.strip() else ""
        for line in text.splitlines()
    )


def _print_email(label: str, text: str) -> None:
    print(f"\n  [{label}]")
    print(_wrap(text[:800]))
    if len(text) > 800:
        print(f"{_INDENT}... ({len(text)} chars total)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview synthetic email generation")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model ID",
    )
    parser.add_argument("--sender", default=None, help="Pin a specific sender ID")
    parser.add_argument("--n-senders", type=int, default=2, help="Number of senders to preview")
    parser.add_argument("--n-examples", type=int, default=5, help="Real emails used as style context")
    parser.add_argument("--n-generate", type=int, default=2, help="Synthetic emails to generate per sender")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantization (needs bitsandbytes)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cfg = load_config(args.config)

    print(f"Loading training split from {cfg.data.processed_dir} ...")
    ds_dict = load_from_disk(cfg.data.processed_dir)
    train_ds = ds_dict["train"]

    sender_to_texts: dict[str, list[str]] = defaultdict(list)
    for text, sid in zip(train_ds["text"], train_ds["sender_id"]):
        sender_to_texts[sid].append(text)

    if args.sender:
        if args.sender not in sender_to_texts:
            available = sorted(sender_to_texts)[:10]
            print(f"Sender '{args.sender}' not found. First 10 available:\n  " + "\n  ".join(available))
            return
        senders = [args.sender]
    else:
        eligible = [s for s, txts in sender_to_texts.items() if len(txts) >= args.n_examples]
        senders = rng.sample(eligible, min(args.n_senders, len(eligible)))

    print(f"Loading model {args.model} ...")
    tokenizer, model = _load_model(args.model, args.load_in_4bit)
    preprocess_cfg = cfg.data.preprocessing

    for sid in senders:
        texts = sender_to_texts[sid]
        examples = rng.sample(texts, args.n_examples)
        examples_truncated = [e[:600] for e in examples]
        topics = rng.sample(_TOPICS, min(args.n_generate, len(_TOPICS)))
        prompts = [_build_prompt(examples_truncated, topic) for topic in topics]

        print(f"\n{'═' * 72}")
        print(f"  SENDER: {sid}  ({len(texts)} real emails)")
        print(f"{'═' * 72}")

        print(f"\n  ── STYLE EXAMPLES (shown to model) ──────────────────────────────")
        for i, ex in enumerate(examples[:3], 1):
            _print_email(f"Real example {i}", ex)

        print(f"\n{_DIVIDER}")
        print(f"  ── GENERATED IMITATIONS ─────────────────────────────────────────")

        generated = _generate_batch(prompts, tokenizer, model)

        for i, (raw, topic) in enumerate(zip(generated, topics), 1):
            cleaned = preprocess(raw, preprocess_cfg)
            status = "ACCEPTED" if cleaned is not None else "REJECTED by preprocessor"
            print(f"\n  Topic given: \"{topic}\"  →  {status}")
            _print_email(f"Generated {i}", cleaned if cleaned is not None else raw)

        print(f"\n{_DIVIDER}")

    print("\nDone.")


if __name__ == "__main__":
    main()

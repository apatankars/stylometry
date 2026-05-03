"""Generate synthetic hard-negative emails for each training sender.

For each sender in the training split, sample a few real emails as style context,
then prompt a local LLM to write new emails that mimic that sender's voice but on
unrelated topics.  The resulting Arrow dataset is consumed by
SyntheticAugmentedDataset + SyntheticBalancedSampler during training.

Each synthetic email is run through the same preprocessing pipeline as the real
data so token distributions match.

Usage:
    python scripts/generate_synthetic_emails.py \\
        --config configs/base.yaml \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --n-per-sender 10 \\
        --n-examples 5 \\
        --output data/synthetic/enron_synthetic \\
        --load-in-4bit

Model recommendation:
    mistralai/Mistral-7B-Instruct-v0.3  -- strong style following, ~7 GB VRAM (4-bit)
    meta-llama/Llama-3.1-8B-Instruct    -- close second, same VRAM

Requirements:
    pip install transformers accelerate bitsandbytes datasets
    (bitsandbytes only needed with --load-in-4bit)
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from email_fraud.config import load_config
from email_fraud.data.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Diverse topics drawn from business/professional email contexts.
# Sampled randomly per generation to prevent topic leakage from style examples.
# ---------------------------------------------------------------------------
_TOPICS = [
    "a scheduling conflict for next week",
    "a quarterly budget update",
    "an equipment or software request",
    "a team lunch or social event",
    "following up on an overdue invoice",
    "requesting feedback on a document",
    "a data access or permissions issue",
    "travel arrangements for a conference",
    "a vendor contract renewal",
    "a new hire starting on the team",
    "an upcoming office renovation",
    "a client escalation that needs handling",
    "a project milestone being delayed",
    "a policy change or HR announcement",
    "an IT outage or system maintenance window",
    "congratulating a colleague on a promotion",
    "clarifying action items from a meeting",
    "asking about the status of a pending report",
    "a parking or building access issue",
    "a last-minute change to a deliverable",
]


def _build_prompt(examples: list[str], topic: str) -> str:
    joined = "\n---\n".join(examples)
    return (
        "You are imitating the writing style of one person based on their emails.\n\n"
        f"Here are examples of their writing:\n---\n{joined}\n---\n\n"
        "Write a new email they might send. Faithfully reproduce:\n"
        "- Sentence length and rhythm\n"
        "- Punctuation habits (comma splices, ellipses, run-ons, capitalization quirks)\n"
        "- Greeting and sign-off style\n"
        "- Level of formality and hedging language\n"
        "- Any characteristic vocabulary, filler phrases, or personal quirks\n\n"
        f"Write about: {topic}\n"
        "Do NOT copy any sentences from the examples above.\n"
        "Output only the email body. No subject line, no metadata."
    )


def _load_model(model_name: str, load_in_4bit: bool):
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )
    model.eval()
    return tokenizer, model


def _generate_batch(
    prompts: list[str],
    tokenizer,
    model,
    max_new_tokens: int = 300,
    temperature: float = 0.85,
    top_p: float = 0.9,
) -> list[str]:
    """Run a list of prompts through the model and return generated texts."""
    # Format as chat messages using the model's template.
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, output_ids in enumerate(outputs):
        # Slice off the prompt tokens to get only the generated portion.
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append(text)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic hard-negative emails")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model ID (default: Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument("--n-per-sender", type=int, default=10, help="Synthetic emails to generate per sender")
    parser.add_argument("--n-examples", type=int, default=5, help="Real emails to use as style context")
    parser.add_argument("--output", required=True, help="Output path for Arrow dataset")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (needs bitsandbytes)")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts to process in parallel")
    parser.add_argument("--max-senders", type=int, default=None, help="Limit to N senders (for debugging)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cfg = load_config(args.config)

    print(f"Loading training split from {cfg.data.processed_dir}")
    ds_dict = load_from_disk(cfg.data.processed_dir)
    train_ds = ds_dict["train"]

    # Build sender → list of texts index.
    sender_to_texts: dict[str, list[str]] = defaultdict(list)
    for text, sid in zip(train_ds["text"], train_ds["sender_id"]):
        sender_to_texts[sid].append(text)

    senders = sorted(sender_to_texts.keys())
    if args.max_senders:
        senders = senders[: args.max_senders]

    print(f"Found {len(senders)} senders. Loading model {args.model}...")
    tokenizer, model = _load_model(args.model, args.load_in_4bit)

    syn_texts: list[str] = []
    syn_sender_ids: list[str] = []
    syn_source_ids: list[str] = []

    preprocess_cfg = cfg.data.preprocessing

    for sender_idx, sid in enumerate(senders):
        texts = sender_to_texts[sid]
        if len(texts) < args.n_examples:
            # Skip senders with too few emails to build a good style profile.
            continue

        syn_sid = f"{sid}__syn"
        examples = rng.sample(texts, args.n_examples)
        # Truncate examples so the prompt stays within the model's context window.
        examples = [e[:600] for e in examples]

        # Build one prompt per desired synthetic email with a unique random topic.
        topics = rng.choices(_TOPICS, k=args.n_per_sender)
        prompts = [_build_prompt(examples, topic) for topic in topics]

        generated: list[str] = []
        for batch_start in range(0, len(prompts), args.batch_size):
            batch = prompts[batch_start : batch_start + args.batch_size]
            generated.extend(_generate_batch(batch, tokenizer, model))

        # Run each generated email through the standard preprocessing pipeline.
        # This filters out degenerate outputs and normalises whitespace/length.
        accepted = 0
        for raw in generated:
            cleaned = preprocess(raw, preprocess_cfg)
            if cleaned is not None:
                syn_texts.append(cleaned)
                syn_sender_ids.append(syn_sid)
                syn_source_ids.append(sid)
                accepted += 1

        print(
            f"[{sender_idx + 1}/{len(senders)}] {sid}: "
            f"{accepted}/{args.n_per_sender} accepted after preprocessing"
        )

    print(f"\nTotal synthetic emails: {len(syn_texts)}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_ds = Dataset.from_dict(
        {
            "text": syn_texts,
            "sender_id": syn_sender_ids,
            "source_sender_id": syn_source_ids,
        }
    )
    synthetic_ds.save_to_disk(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

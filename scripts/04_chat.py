"""Interactive chat with a scratch checkpoint.

This is a lightweight local CLI for inspecting a trained checkpoint. A model
trained only with causal LM pretraining will not reliably behave like an
assistant; SFT data is needed for instruction-following behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scratch_llm.config import GenerationConfig, ModelConfig
from scratch_llm.inference import generate
from scratch_llm.inference.config_loader import build_model_config_from_checkpoint
from scratch_llm.model import ScratchLLM
from scratch_llm.tokenizer import load_tokenizer
from scratch_llm.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse interactive chat CLI arguments."""

    parser = argparse.ArgumentParser(description="Chat with scratch_llm")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer-dir", default="scratch_llm_runs/tokenizer", help="Tokenizer directory")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 means greedy decoding")
    parser.add_argument("--top-k", type=int, default=50, help="Optional top-k sampling")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocabulary size")
    parser.add_argument("--dim", type=int, default=None, help="Override model hidden size")
    parser.add_argument("--n-layers", type=int, default=None, help="Override number of decoder blocks")
    parser.add_argument("--n-heads", type=int, default=None, help="Override number of attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Override number of KV heads")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override context length")
    parser.add_argument(
        "--plain-prompt",
        action="store_true",
        help="Do not use the tokenizer chat template",
    )
    return parser.parse_args()


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    """Build ModelConfig from checkpoint metadata plus explicit CLI overrides."""

    overrides = {
        "vocab_size": args.vocab_size,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "max_seq_len": args.max_seq_len,
    }
    return build_model_config_from_checkpoint(args.checkpoint, overrides)


def format_prompt(tokenizer: object, messages: list[dict[str, str]], plain_prompt: bool) -> str:
    """Format the running dialogue as text for the model."""

    if not plain_prompt and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    chunks: list[str] = []
    for message in messages:
        chunks.append(f"{message['role']}: {message['content']}")
    chunks.append("assistant:")
    return "\n".join(chunks)


def main() -> None:
    """Run an interactive prompt loop."""

    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model_config = build_model_config(args)
    model = ScratchLLM(model_config).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
    messages: list[dict[str, str]] = []

    print("Type 'exit' or 'quit' to stop.", flush=True)
    while True:
        user_text = input("user> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        prompt = format_prompt(tokenizer, messages, args.plain_prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        input_ids = input_ids[:, -model_config.max_seq_len :]

        new_ids = generate(model, input_ids, gen_config)
        assistant_text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        print(f"assistant> {assistant_text}", flush=True)
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()

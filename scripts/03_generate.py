"""Generate text from a scratch checkpoint.

Run after you implement checkpoint loading and generation.
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
from scratch_llm.inference.config_loader import build_model_config_from_checkpoint
from scratch_llm.inference import generate
from scratch_llm.model import ScratchLLM
from scratch_llm.tokenizer import load_tokenizer
from scratch_llm.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse generation CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate with scratch_llm")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer-dir", default="scratch_llm_runs/tokenizer", help="Tokenizer directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="0 means greedy decoding")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocabulary size")
    parser.add_argument("--dim", type=int, default=None, help="Override model hidden size")
    parser.add_argument("--n-layers", type=int, default=None, help="Override number of decoder blocks")
    parser.add_argument("--n-heads", type=int, default=None, help="Override number of attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Override number of KV heads")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override context length")
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


def main() -> None:
    """Load model and tokenizer, then print decoded generation."""

    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model_config = build_model_config(args)
    model = ScratchLLM(model_config).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    encoded = tokenizer(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor(
        [encoded["input_ids"]],
        dtype=torch.long,
        device=args.device,
    )
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
    new_ids = generate(model, input_ids, gen_config)
    text = tokenizer.decode(torch.cat([input_ids, new_ids], dim=1)[0], skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()

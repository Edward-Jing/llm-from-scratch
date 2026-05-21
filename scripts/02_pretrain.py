"""Pretrain the scratch decoder-only LLM.

Run after you implement the tokenizer, dataset, model, and training loop.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scratch_llm.config import ModelConfig, TrainConfig
from scratch_llm.data import CausalLMDataset
from scratch_llm.model import ScratchLLM
from scratch_llm.tokenizer import load_tokenizer
from scratch_llm.training import save_checkpoint, train_one_epoch
from scratch_llm.utils import count_parameters, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse pretraining CLI arguments."""

    parser = argparse.ArgumentParser(description="Pretrain scratch_llm")
    parser.add_argument("--data-path", required=True, help="JSONL pretraining data path")
    parser.add_argument("--tokenizer-dir", default="scratch_llm_runs/tokenizer", help="Tokenizer directory")
    parser.add_argument("--output-dir", default="scratch_llm_runs/checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Autocast precision",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Context length")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches per optimizer update",
    )
    parser.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument("--log-interval", type=int, default=10, help="Training log interval")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dim", type=int, default=512, help="Model hidden size")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of decoder blocks")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Number of KV heads")
    parser.add_argument("--vocab-size", type=int, default=6144, help="Tokenizer vocabulary size")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--save-name", default="final.pt", help="Final checkpoint file name")
    return parser.parse_args()


def log_metrics(metrics: dict[str, object]) -> None:
    """Print compact training metrics."""

    formatted = " ".join(f"{key}={value}" for key, value in metrics.items())
    print(formatted, flush=True)


def main() -> None:
    """Build model, dataset, optimizer, and run pretraining."""

    args = parse_args()
    seed_everything(args.seed)

    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        data_path=args.data_path,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(train_config.tokenizer_dir)
    dataset = CausalLMDataset(
        train_config.data_path,
        tokenizer,
        max_length=train_config.max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    model = ScratchLLM(model_config).to(train_config.device)
    print(f"trainable_parameters={count_parameters(model):,}", flush=True)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    updates_per_epoch = math.ceil(max(len(dataloader), 1) / max(train_config.accumulation_steps, 1))
    total_steps = args.epochs * updates_per_epoch
    step = 0
    for epoch in range(train_config.epochs):
        step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            config=train_config,
            epoch=epoch,
            total_steps=total_steps,
            start_step=step,
            logger=log_metrics,
        )
        checkpoint_extra = {
            "epoch": epoch,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        }
        epoch_path = output_dir / f"epoch_{epoch + 1}.pt"
        save_checkpoint(epoch_path, model, optimizer=optimizer, step=step, extra=checkpoint_extra)
        print(f"saved_checkpoint={epoch_path}", flush=True)

    final_path = output_dir / args.save_name
    save_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        step=step,
        extra={
            "epoch": train_config.epochs - 1,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
    )
    print(f"final_checkpoint={final_path}", flush=True)


if __name__ == "__main__":
    main()

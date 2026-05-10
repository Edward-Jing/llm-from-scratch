# Scratch LLM Package

This package contains the learning scaffold for implementing a small decoder-only LLM from scratch. It defines the module layout, function signatures, input/output contracts, and TODO markers that guide the implementation process.

## Reference

The learning path follows the original Happy-LLM project:

- Original repository: [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)
- Online guide: [Happy-LLM documentation](https://datawhalechina.github.io/happy-llm/)

This package is a personal learning scaffold. It is not a copy of the original tutorial implementation.

## Goals

1. Train a ByteLevel BPE tokenizer from JSONL text.
2. Build a causal language modeling dataset.
3. Implement a LLaMA-style decoder-only Transformer by hand.
4. Implement pretraining, checkpointing, and generation.
5. Keep progress easy to test, commit, and review.

## Layout

```text
scratch_llm/
  config.py              # Project-wide dataclass configs
  tokenizer.py           # BPE tokenizer workflow
  data/
    jsonl.py             # JSONL readers
    dataset.py           # Dataset construction
  model/
    norm.py              # RMSNorm
    rope.py              # Rotary position embeddings
    attention.py         # Causal self-attention / grouped-query attention
    mlp.py               # SwiGLU feed-forward layer
    blocks.py            # Decoder block
    transformer.py       # Full language model
  training/
    lr.py                # Learning-rate schedule
    loop.py              # Training and evaluation loops
    checkpoint.py        # Checkpoint helpers
  inference/
    generation.py        # Sampling and generation
  utils/
    seed.py              # Reproducibility helpers
    params.py            # Parameter counting
```

## Implementation Order

1. `scratch_llm/data/jsonl.py`
2. `scratch_llm/tokenizer.py`
3. `scratch_llm/data/dataset.py`
4. `scratch_llm/model/norm.py`
5. `scratch_llm/model/rope.py`
6. `scratch_llm/model/mlp.py`
7. `scratch_llm/model/attention.py`
8. `scratch_llm/model/blocks.py`
9. `scratch_llm/model/transformer.py`
10. `scratch_llm/training/lr.py`
11. `scratch_llm/training/loop.py`
12. `scratch_llm/training/checkpoint.py`
13. `scratch_llm/inference/generation.py`

## Local Checks

```bash
python3 -m unittest tests/test_contracts.py
python3 -m unittest -v tests/test_jsonl.py
python3 -m compileall scratch_llm scripts tests
```

## Script Entry Points

```bash
python3 scripts/01_train_tokenizer.py --data-path data/pretrain.jsonl
python3 scripts/02_pretrain.py --data-path data/pretrain.jsonl --device cpu
python3 scripts/03_generate.py --checkpoint scratch_llm_runs/checkpoints/model.pt --prompt "Hello"
```

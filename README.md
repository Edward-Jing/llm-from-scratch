# LLM From Scratch

This is Edward's learning project for implementing a small decoder-only LLM from scratch while studying the Happy-LLM guide. The goal is not to copy a finished implementation, but to build each function step by step with clear module boundaries, function signatures, tests, and notes.

## Reference

This project is inspired by and follows the learning path of the original Happy-LLM project:

- Original repository: [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)
- Online guide: [Happy-LLM documentation](https://datawhalechina.github.io/happy-llm/)

Please cite or link the original Happy-LLM project when sharing this learning project, because the architecture and study order are based on that guide.

## Goals

1. Train a ByteLevel BPE tokenizer from JSONL text data.
2. Build a causal language modeling dataset.
3. Implement a LLaMA-style decoder-only Transformer by hand.
4. Implement the pretraining loop, checkpointing, and text generation.
5. Keep local progress synchronized with GitHub through regular commits and pushes.

## Project Layout

```text
scratch_llm/
  config.py              # ModelConfig / TokenizerConfig / TrainConfig / GenerationConfig
  tokenizer.py           # BPE tokenizer training, saving, loading, and validation
  data/
    jsonl.py             # JSONL readers
    dataset.py           # Pretraining/SFT datasets and loss masks
  model/
    norm.py              # RMSNorm
    rope.py              # Rotary position embeddings
    attention.py         # Causal self-attention / grouped-query attention
    mlp.py               # SwiGLU feed-forward network
    blocks.py            # DecoderBlock
    transformer.py       # ScratchLLM
  training/
    lr.py                # Warmup + cosine decay schedule
    loop.py              # Training and evaluation loops
    checkpoint.py        # Checkpoint save/load helpers
  inference/
    generation.py        # Top-k sampling and autoregressive generation
  utils/
    seed.py              # Random seed helpers
    params.py            # Parameter counting
scripts/
  01_train_tokenizer.py
  02_pretrain.py
  03_generate.py
tests/
  test_contracts.py
  test_jsonl.py
  jsonl_cases/
Practice by Parts/
  Practice_dataclass.md
```

## Recommended Implementation Order

Start with pure Python helpers, then move to tensor operations, then training.

1. `scratch_llm/data/jsonl.py` 5.10 
2. `scratch_llm/tokenizer.py` 5.12
3. `scratch_llm/data/dataset.py` 5.13
4. `scratch_llm/model/norm.py` 5.13
5. `scratch_llm/model/rope.py` 5.14
6. `scratch_llm/model/mlp.py` 5.14
7. `scratch_llm/model/attention.py` 5.20
8. `scratch_llm/model/blocks.py`5.20
9. `scratch_llm/model/transformer.py` 5.20
10. `scratch_llm/training/lr.py` 5.20
11. `scratch_llm/training/loop.py` 5.20
12. `scratch_llm/training/checkpoint.py` 5.20
13. `scratch_llm/inference/generation.py` 5.20

## Local Checks

Run the interface contract tests:

```bash
python3 -m unittest tests/test_contracts.py
```

Run the JSONL unit tests:

```bash
python3 -m unittest -v tests/test_jsonl.py
```

Run syntax checks:

```bash
python3 -m compileall scratch_llm scripts tests
```

After implementations are filled in, the project should support:

```bash
python3 scripts/01_train_tokenizer.py --data-path data/pretrain.jsonl
python3 scripts/02_pretrain.py --data-path data/pretrain.jsonl --device cpu
python3 scripts/03_generate.py --checkpoint scratch_llm_runs/checkpoints/model.pt --prompt "Hello"
```

## GitHub Workflow

Work locally first, then commit and push regularly:

```bash
git status
git add .
git commit -m "Implement JSONL helpers"
git push
```

In GitHub Desktop, the same workflow is:

1. Edit files locally.
2. Review changes in the Changes tab.
3. Write a short commit summary.
4. Click `Commit to scratch-llm-starter`.
5. Click `Push origin`.

## Practice Notes

Some supporting Python knowledge is practiced in `Practice by Parts/`. The current practice topic is:

1. `@dataclass(slots=True)` and basic object-oriented programming.

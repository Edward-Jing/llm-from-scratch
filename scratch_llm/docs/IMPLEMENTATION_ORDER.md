# Implementation Order

This order follows the practical learning path of the original [Happy-LLM](https://github.com/datawhalechina/happy-llm) guide, but keeps the work scoped to this personal scratch implementation.

## Step 1: Data Reading

Implement `iter_jsonl_records`, `iter_jsonl_texts`, and `count_jsonl_lines` first. This step only needs the standard library: `json`, `pathlib`, and file I/O. No tensor work is required yet.

Check:

```bash
python3 -m unittest tests/test_contracts.py
python3 -m unittest -v tests/test_jsonl.py
```

## Step 2: Tokenizer

Implement `train_bpe_tokenizer`, `write_tokenizer_configs`, `load_tokenizer`, and `validate_tokenizer`. Use the Happy-LLM tokenizer workflow as conceptual guidance, then write the implementation yourself.

Minimal data format:

```jsonl
{"text": "hello world"}
{"text": "language model learns next tokens"}
```

## Step 3: Dataset

Implement `build_causal_lm_example` first, then implement `CausalLMDataset`.

Important shapes:

```text
unshifted input_ids: max_length
x: max_length - 1
y: max_length - 1
loss_mask: max_length - 1
```

## Step 4: Small Model Modules

Recommended order:

1. `RMSNorm`
2. `precompute_rope_frequencies`
3. `reshape_for_broadcast`
4. `apply_rotary_embedding`
5. `derive_swiglu_hidden_dim`
6. `SwiGLU`
7. `repeat_kv`

Each function can be tested with small random tensors before any training loop exists.

## Step 5: Attention And Decoder Block

First implement causal attention without a padding mask. Then add `attention_mask` support.

Shape guide:

```text
x:  (batch, seq_len, dim)
q:  (batch, seq_len, n_heads, head_dim)
k/v:(batch, seq_len, n_kv_heads, head_dim)
attention input after transpose: (batch, heads, seq_len, head_dim)
output: (batch, seq_len, dim)
```

## Step 6: ScratchLLM

Connect token embeddings, decoder blocks, final RMSNorm, and the LM head. During training, return full logits and loss. For generation, start with a simple no-KV-cache implementation.

## Step 7: Training Loop

Make a tiny CPU model work first. GPU, mixed precision, and multi-GPU training can come later.

Minimal run:

```bash
python3 scripts/02_pretrain.py \
  --data-path data/pretrain.jsonl \
  --device cpu \
  --dim 128 \
  --n-layers 2 \
  --n-heads 4 \
  --batch-size 2 \
  --max-seq-len 64
```

## Step 8: Generation

Start with greedy decoding using `temperature=0`. After that works, add `top_k` and random sampling.

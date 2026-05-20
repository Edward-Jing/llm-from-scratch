"""Unit tests and visual checks for scratch_llm.model.attention."""

from __future__ import annotations

import math
import unittest

import torch

from scratch_llm.config import ModelConfig
from scratch_llm.model.attention import CausalSelfAttention, repeat_kv
from scratch_llm.model.rope import apply_rotary_embedding, precompute_rope_frequencies


def reference_attention(
    module: CausalSelfAttention,
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference attention implementation written directly from the math."""

    batch_size, seq_len, _ = x.shape

    q = module.wq(x).view(batch_size, seq_len, module.n_heads, module.head_dim)
    k = module.wk(x).view(batch_size, seq_len, module.n_kv_heads, module.head_dim)
    v = module.wv(x).view(batch_size, seq_len, module.n_kv_heads, module.head_dim)

    q, k = apply_rotary_embedding(q, k, freqs_cos, freqs_sin)
    k = repeat_kv(k, module.n_rep)
    v = repeat_kv(v, module.n_rep)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(module.head_dim)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
    ).view(1, 1, seq_len, seq_len)
    mask_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~causal_mask, mask_value)

    if attention_mask is not None:
        key_mask = attention_mask.to(device=x.device, dtype=torch.bool).view(
            batch_size, 1, 1, seq_len
        )
        scores = scores.masked_fill(~key_mask, mask_value)

    attn_probs = torch.softmax(scores.float(), dim=-1).type_as(scores)
    out = torch.matmul(attn_probs, v)
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, module.dim)
    return module.wo(out)


def format_token_rows(name: str, tensor: torch.Tensor) -> str:
    """Format a (batch, seq, dim) tensor as readable token rows."""

    rows = [f"{name}:"]
    data = tensor.detach().cpu()
    for batch_idx in range(data.shape[0]):
        for token_idx in range(data.shape[1]):
            values = ", ".join(f"{v:8.4f}" for v in data[batch_idx, token_idx].tolist())
            rows.append(f"  batch={batch_idx}, token={token_idx}: [{values}]")
    return "\n".join(rows)


class RepeatKVTest(unittest.TestCase):
    def test_repeat_kv_returns_input_when_repeat_is_one(self) -> None:
        x = torch.randn(2, 3, 4, 5)

        actual = repeat_kv(x, n_rep=1)

        self.assertIs(actual, x)

    def test_repeat_kv_repeats_each_key_value_head(self) -> None:
        x = torch.tensor(
            [
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[3.0, 30.0], [4.0, 40.0]],
                ]
            ]
        )

        actual = repeat_kv(x, n_rep=3)
        expected = torch.tensor(
            [
                [
                    [
                        [1.0, 10.0],
                        [1.0, 10.0],
                        [1.0, 10.0],
                        [2.0, 20.0],
                        [2.0, 20.0],
                        [2.0, 20.0],
                    ],
                    [
                        [3.0, 30.0],
                        [3.0, 30.0],
                        [3.0, 30.0],
                        [4.0, 40.0],
                        [4.0, 40.0],
                        [4.0, 40.0],
                    ],
                ]
            ]
        )

        self.assertEqual(tuple(actual.shape), (1, 2, 6, 2))
        self.assertTrue(torch.equal(actual, expected))


class CausalSelfAttentionTest(unittest.TestCase):
    def make_attention(
        self,
        dim: int = 8,
        n_heads: int = 4,
        n_kv_heads: int | None = 2,
        max_seq_len: int = 6,
    ) -> CausalSelfAttention:
        config = ModelConfig(
            vocab_size=32,
            dim=dim,
            n_layers=1,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            dropout=0.0,
        )
        module = CausalSelfAttention(config)
        module.eval()
        return module

    def test_forward_returns_expected_shape(self) -> None:
        torch.manual_seed(0)
        module = self.make_attention()
        x = torch.randn(2, 5, 8)
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            head_dim=module.head_dim,
            max_seq_len=module.config.max_seq_len,
        )

        actual = module(x, freqs_cos, freqs_sin)

        self.assertEqual(tuple(actual.shape), (2, 5, 8))

    def test_forward_matches_reference_implementation_with_gqa(self) -> None:
        torch.manual_seed(1)
        module = self.make_attention(dim=8, n_heads=4, n_kv_heads=2, max_seq_len=5)
        x = torch.randn(2, 4, 8)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            head_dim=module.head_dim,
            max_seq_len=module.config.max_seq_len,
        )

        actual = module(x, freqs_cos, freqs_sin, attention_mask)
        expected = reference_attention(module, x, freqs_cos, freqs_sin, attention_mask)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_causal_mask_blocks_future_tokens(self) -> None:
        torch.manual_seed(2)
        module = self.make_attention(dim=8, n_heads=2, n_kv_heads=None, max_seq_len=4)
        x = torch.randn(1, 4, 8)
        x_future_changed = x.clone()
        x_future_changed[:, 1:] = torch.randn(1, 3, 8) * 50.0
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            head_dim=module.head_dim,
            max_seq_len=module.config.max_seq_len,
        )

        out = module(x, freqs_cos, freqs_sin)
        out_future_changed = module(x_future_changed, freqs_cos, freqs_sin)

        self.assertTrue(torch.allclose(out[:, 0], out_future_changed[:, 0], atol=1e-6))

    def test_attention_mask_blocks_padded_keys(self) -> None:
        torch.manual_seed(3)
        module = self.make_attention(dim=8, n_heads=2, n_kv_heads=None, max_seq_len=3)
        x = torch.randn(1, 3, 8)
        x_masked_token_changed = x.clone()
        x_masked_token_changed[:, 1] = torch.randn(1, 8) * 100.0
        attention_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            head_dim=module.head_dim,
            max_seq_len=module.config.max_seq_len,
        )

        out = module(x, freqs_cos, freqs_sin, attention_mask)
        out_masked_token_changed = module(
            x_masked_token_changed,
            freqs_cos,
            freqs_sin,
            attention_mask,
        )

        self.assertTrue(torch.allclose(out[:, 2], out_masked_token_changed[:, 2], atol=1e-6))

    def test_forward_rejects_invalid_shapes(self) -> None:
        module = self.make_attention(dim=8, n_heads=2, n_kv_heads=None, max_seq_len=3)
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            head_dim=module.head_dim,
            max_seq_len=module.config.max_seq_len,
        )

        with self.assertRaisesRegex(ValueError, "Expected x.shape"):
            module(torch.randn(1, 2, 6), freqs_cos, freqs_sin)

        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            module(torch.randn(1, 4, 8), freqs_cos, freqs_sin)

        with self.assertRaisesRegex(ValueError, "attention_mask must have shape"):
            module(
                torch.randn(1, 2, 8),
                freqs_cos,
                freqs_sin,
                attention_mask=torch.ones(1, 3),
            )

    def test_visual_comparison_for_learning(self) -> None:
        torch.manual_seed(4)
        module = self.make_attention(dim=4, n_heads=1, n_kv_heads=None, max_seq_len=3)
        x = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        )
        x_future_changed = x.clone()
        x_future_changed[:, 1:] = torch.tensor(
            [
                [
                    [10.0, 10.0, 10.0, 10.0],
                    [-5.0, -5.0, -5.0, -5.0],
                ]
            ]
        )
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            head_dim=module.head_dim,
            max_seq_len=module.config.max_seq_len,
        )

        out = module(x, freqs_cos, freqs_sin)
        out_future_changed = module(x_future_changed, freqs_cos, freqs_sin)

        print("\n=== CausalSelfAttention Visual Check ===")
        print(format_token_rows("input", x))
        print(format_token_rows("future changed input", x_future_changed))
        print(format_token_rows("output", out))
        print(format_token_rows("output after future tokens changed", out_future_changed))
        print("token 0 is unchanged because the causal mask blocks future tokens")

        self.assertTrue(torch.allclose(out[:, 0], out_future_changed[:, 0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

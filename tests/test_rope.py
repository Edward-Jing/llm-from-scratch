"""Unit tests for scratch_llm.model.rope."""

from __future__ import annotations

import unittest

import torch

from scratch_llm.model.rope import (
    apply_rotary_embedding,
    precompute_rope_frequencies,
    reshape_for_broadcast,
)


def reference_apply_rope(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Reference RoPE implementation using explicit even/odd coordinates."""

    seq_len = x.shape[1]
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    cos = freqs_cos[:seq_len].view(1, seq_len, 1, x_even.shape[-1])
    sin = freqs_sin[:seq_len].view(1, seq_len, 1, x_even.shape[-1])

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).type_as(x)


def format_rope_rows(name: str, tensor: torch.Tensor) -> str:
    """Format a (batch, seq, heads, head_dim) tensor for visual inspection."""

    rows = [f"{name}:"]
    data = tensor.detach().cpu()
    for batch_idx in range(data.shape[0]):
        for token_idx in range(data.shape[1]):
            for head_idx in range(data.shape[2]):
                values = ", ".join(f"{v:8.4f}" for v in data[batch_idx, token_idx, head_idx].tolist())
                rows.append(
                    f"  batch={batch_idx}, token={token_idx}, head={head_idx}: [{values}]"
                )
    return "\n".join(rows)


class RoPEFrequencyTest(unittest.TestCase):
    def test_precompute_rope_frequencies_shape_and_position_zero(self) -> None:
        cos, sin = precompute_rope_frequencies(head_dim=6, max_seq_len=4)

        self.assertEqual(tuple(cos.shape), (4, 3))
        self.assertEqual(tuple(sin.shape), (4, 3))
        self.assertTrue(torch.allclose(cos[0], torch.ones(3)))
        self.assertTrue(torch.allclose(sin[0], torch.zeros(3)))

    def test_precompute_rope_frequencies_matches_manual_formula(self) -> None:
        head_dim = 4
        theta = 10000.0
        cos, sin = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=3, theta=theta)

        inv_freq = torch.tensor([1.0, 1.0 / (theta ** (2.0 / head_dim))])
        manual_freqs = torch.outer(torch.arange(3).float(), inv_freq)

        self.assertTrue(torch.allclose(cos, torch.cos(manual_freqs)))
        self.assertTrue(torch.allclose(sin, torch.sin(manual_freqs)))

    def test_precompute_rope_frequencies_rejects_odd_head_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "head_dim must be even"):
            precompute_rope_frequencies(head_dim=5, max_seq_len=4)


class RoPEBroadcastTest(unittest.TestCase):
    def test_reshape_for_broadcast_returns_expected_view_shape(self) -> None:
        freqs = torch.ones(3, 2)
        x_half = torch.zeros(2, 3, 4, 2)

        reshaped = reshape_for_broadcast(freqs, x_half)

        self.assertEqual(tuple(reshaped.shape), (1, 3, 1, 2))

    def test_reshape_for_broadcast_rejects_shape_mismatch(self) -> None:
        freqs = torch.ones(4, 2)
        x_half = torch.zeros(2, 3, 4, 2)

        with self.assertRaisesRegex(ValueError, "freqs must have shape"):
            reshape_for_broadcast(freqs, x_half)


class ApplyRotaryEmbeddingTest(unittest.TestCase):
    def test_apply_rotary_embedding_matches_reference(self) -> None:
        q = torch.tensor(
            [
                [
                    [[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.0, -1.0]],
                    [[5.0, 6.0, 7.0, 8.0], [2.0, 0.0, 0.0, 2.0]],
                    [[9.0, 10.0, 11.0, 12.0], [-1.0, 1.0, -2.0, 2.0]],
                ]
            ]
        )
        k = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0, 1.0]],
                    [[2.0, 1.0, 1.0, 2.0]],
                    [[3.0, 2.0, 2.0, 3.0]],
                ]
            ]
        )
        cos, sin = precompute_rope_frequencies(head_dim=4, max_seq_len=3)

        actual_q, actual_k = apply_rotary_embedding(q, k, cos, sin)
        expected_q = reference_apply_rope(q, cos, sin)
        expected_k = reference_apply_rope(k, cos, sin)

        self.assertEqual(actual_q.shape, q.shape)
        self.assertEqual(actual_k.shape, k.shape)
        self.assertTrue(torch.allclose(actual_q, expected_q, atol=1e-6))
        self.assertTrue(torch.allclose(actual_k, expected_k, atol=1e-6))

    def test_apply_rotary_embedding_position_zero_is_identity(self) -> None:
        q = torch.randn(2, 1, 3, 4)
        k = torch.randn(2, 1, 1, 4)
        cos, sin = precompute_rope_frequencies(head_dim=4, max_seq_len=1)

        actual_q, actual_k = apply_rotary_embedding(q, k, cos, sin)

        self.assertTrue(torch.allclose(actual_q, q, atol=1e-6))
        self.assertTrue(torch.allclose(actual_k, k, atol=1e-6))

    def test_apply_rotary_embedding_preserves_pair_norms(self) -> None:
        q = torch.randn(2, 5, 3, 6)
        k = torch.randn(2, 5, 2, 6)
        cos, sin = precompute_rope_frequencies(head_dim=6, max_seq_len=5)

        actual_q, actual_k = apply_rotary_embedding(q, k, cos, sin)

        q_pair_norm_before = q.float().reshape(*q.shape[:-1], -1, 2).pow(2).sum(dim=-1)
        q_pair_norm_after = actual_q.float().reshape(*q.shape[:-1], -1, 2).pow(2).sum(dim=-1)
        k_pair_norm_before = k.float().reshape(*k.shape[:-1], -1, 2).pow(2).sum(dim=-1)
        k_pair_norm_after = actual_k.float().reshape(*k.shape[:-1], -1, 2).pow(2).sum(dim=-1)

        self.assertTrue(torch.allclose(q_pair_norm_after, q_pair_norm_before, atol=1e-5))
        self.assertTrue(torch.allclose(k_pair_norm_after, k_pair_norm_before, atol=1e-5))

    def test_apply_rotary_embedding_preserves_dtype(self) -> None:
        q = torch.randn(1, 3, 2, 4, dtype=torch.float16)
        k = torch.randn(1, 3, 1, 4, dtype=torch.float16)
        cos, sin = precompute_rope_frequencies(head_dim=4, max_seq_len=3)

        actual_q, actual_k = apply_rotary_embedding(q, k, cos, sin)

        self.assertEqual(actual_q.dtype, torch.float16)
        self.assertEqual(actual_k.dtype, torch.float16)

    def test_apply_rotary_embedding_rejects_odd_head_dim(self) -> None:
        q = torch.randn(1, 3, 2, 5)
        k = torch.randn(1, 3, 1, 5)
        cos = torch.ones(3, 2)
        sin = torch.zeros(3, 2)

        with self.assertRaisesRegex(ValueError, "head_dim must be even"):
            apply_rotary_embedding(q, k, cos, sin)

    def test_visual_comparison_for_learning(self) -> None:
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]])
        k = torch.tensor([[[[1.0, 0.0, 0.0, 1.0]], [[2.0, 1.0, 1.0, 2.0]]]])
        cos, sin = precompute_rope_frequencies(head_dim=4, max_seq_len=2)

        actual_q, actual_k = apply_rotary_embedding(q, k, cos, sin)
        expected_q = reference_apply_rope(q, cos, sin)
        expected_k = reference_apply_rope(k, cos, sin)

        print("\n=== RoPE Visual Comparison ===")
        print(format_rope_rows("q input", q))
        print(format_rope_rows("q expected", expected_q))
        print(format_rope_rows("q actual", actual_q))
        print(format_rope_rows("q abs diff", (actual_q - expected_q).abs()))
        print(format_rope_rows("k input", k))
        print(format_rope_rows("k expected", expected_k))
        print(format_rope_rows("k actual", actual_k))
        print(format_rope_rows("k abs diff", (actual_k - expected_k).abs()))

        self.assertTrue(torch.allclose(actual_q, expected_q, atol=1e-6))
        self.assertTrue(torch.allclose(actual_k, expected_k, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

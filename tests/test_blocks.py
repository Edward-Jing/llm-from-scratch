"""Unit tests and visual checks for scratch_llm.model.blocks."""

from __future__ import annotations

import unittest

import torch

from scratch_llm.config import ModelConfig
from scratch_llm.model.attention import CausalSelfAttention
from scratch_llm.model.blocks import DecoderBlock
from scratch_llm.model.mlp import SwiGLU
from scratch_llm.model.norm import RMSNorm
from scratch_llm.model.rope import precompute_rope_frequencies


def reference_decoder_block(
    block: DecoderBlock,
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference pre-norm decoder block formula."""

    h = x + block.attention(
        block.attention_norm(x),
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        attention_mask=attention_mask,
    )
    return h + block.feed_forward(block.ffn_norm(h))


def format_token_rows(name: str, tensor: torch.Tensor) -> str:
    """Format a (batch, seq, dim) tensor as readable token rows."""

    rows = [f"{name}:"]
    data = tensor.detach().cpu()
    for batch_idx in range(data.shape[0]):
        for token_idx in range(data.shape[1]):
            values = ", ".join(f"{v:8.4f}" for v in data[batch_idx, token_idx].tolist())
            rows.append(f"  batch={batch_idx}, token={token_idx}: [{values}]")
    return "\n".join(rows)


class DecoderBlockTest(unittest.TestCase):
    def make_block(
        self,
        dim: int = 8,
        n_heads: int = 2,
        n_kv_heads: int | None = None,
        hidden_dim: int = 16,
        max_seq_len: int = 5,
    ) -> DecoderBlock:
        config = ModelConfig(
            vocab_size=32,
            dim=dim,
            n_layers=1,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            multiple_of=8,
            max_seq_len=max_seq_len,
            dropout=0.0,
        )
        block = DecoderBlock(layer_id=0, config=config)
        block.eval()
        return block

    def freqs_for(self, block: DecoderBlock) -> tuple[torch.Tensor, torch.Tensor]:
        return precompute_rope_frequencies(
            head_dim=block.config.head_dim,
            max_seq_len=block.config.max_seq_len,
        )

    def test_init_creates_expected_submodules(self) -> None:
        block = self.make_block()

        self.assertEqual(block.layer_id, 0)
        self.assertIsInstance(block.attention, CausalSelfAttention)
        self.assertIsInstance(block.feed_forward, SwiGLU)
        self.assertIsInstance(block.attention_norm, RMSNorm)
        self.assertIsInstance(block.ffn_norm, RMSNorm)

    def test_forward_returns_expected_shape(self) -> None:
        torch.manual_seed(0)
        block = self.make_block()
        freqs_cos, freqs_sin = self.freqs_for(block)
        x = torch.randn(2, 4, block.config.dim)

        actual = block(x, freqs_cos, freqs_sin)

        self.assertEqual(tuple(actual.shape), (2, 4, block.config.dim))

    def test_forward_matches_reference_pre_norm_residual_formula(self) -> None:
        torch.manual_seed(1)
        block = self.make_block(n_kv_heads=1)
        freqs_cos, freqs_sin = self.freqs_for(block)
        x = torch.randn(2, 4, block.config.dim)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )

        actual = block(x, freqs_cos, freqs_sin, attention_mask)
        expected = reference_decoder_block(block, x, freqs_cos, freqs_sin, attention_mask)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_residual_path_returns_input_when_sublayers_are_zero(self) -> None:
        torch.manual_seed(2)
        block = self.make_block()
        freqs_cos, freqs_sin = self.freqs_for(block)
        x = torch.randn(1, 3, block.config.dim)

        with torch.no_grad():
            for parameter in block.attention.parameters():
                parameter.zero_()
            for parameter in block.feed_forward.parameters():
                parameter.zero_()

        actual = block(x, freqs_cos, freqs_sin)

        self.assertTrue(torch.allclose(actual, x, atol=1e-6))

    def test_attention_mask_blocks_cross_token_effects_from_masked_keys(self) -> None:
        torch.manual_seed(3)
        block = self.make_block(max_seq_len=3)
        freqs_cos, freqs_sin = self.freqs_for(block)
        x = torch.randn(1, 3, block.config.dim)
        x_masked_token_changed = x.clone()
        x_masked_token_changed[:, 1] = torch.randn(1, block.config.dim) * 100.0
        attention_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

        out = block(x, freqs_cos, freqs_sin, attention_mask)
        out_masked_token_changed = block(
            x_masked_token_changed,
            freqs_cos,
            freqs_sin,
            attention_mask,
        )

        self.assertTrue(torch.allclose(out[:, 2], out_masked_token_changed[:, 2], atol=1e-6))

    def test_visual_comparison_for_learning(self) -> None:
        torch.manual_seed(4)
        block = self.make_block(dim=4, n_heads=1, hidden_dim=8, max_seq_len=3)
        freqs_cos, freqs_sin = self.freqs_for(block)
        x = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        )

        actual = block(x, freqs_cos, freqs_sin)
        expected = reference_decoder_block(block, x, freqs_cos, freqs_sin)
        diff = (actual - expected).abs()

        print("\n=== DecoderBlock Visual Comparison ===")
        print(format_token_rows("input", x))
        print(format_token_rows("expected", expected))
        print(format_token_rows("actual", actual))
        print(format_token_rows("abs diff", diff))
        print(f"max_abs_diff: {diff.max().item():.8f}")

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

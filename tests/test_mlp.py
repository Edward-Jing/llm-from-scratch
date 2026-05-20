"""Unit tests and visual checks for scratch_llm.model.mlp."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from scratch_llm.model.mlp import SwiGLU, derive_swiglu_hidden_dim


def reference_swiglu(module: SwiGLU, x: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU formula using the module's own weights."""

    gate = F.silu(module.w1(x))
    up = module.w3(x)
    return module.w2(gate * up)


def format_token_rows(name: str, tensor: torch.Tensor) -> str:
    """Format a (batch, seq, dim) tensor as readable token rows."""

    rows = [f"{name}:"]
    data = tensor.detach().cpu()
    for batch_idx in range(data.shape[0]):
        for token_idx in range(data.shape[1]):
            values = ", ".join(f"{v:8.4f}" for v in data[batch_idx, token_idx].tolist())
            rows.append(f"  batch={batch_idx}, token={token_idx}: [{values}]")
    return "\n".join(rows)


class SwiGLUHiddenDimTest(unittest.TestCase):
    def test_explicit_hidden_dim_is_returned_unchanged(self) -> None:
        actual = derive_swiglu_hidden_dim(dim=512, hidden_dim=123, multiple_of=64)

        self.assertEqual(actual, 123)

    def test_hidden_dim_uses_llama_swiglu_rounding_rule(self) -> None:
        actual = derive_swiglu_hidden_dim(dim=512, hidden_dim=None, multiple_of=64)

        self.assertEqual(actual, 1408)

    def test_hidden_dim_rounds_up_to_multiple(self) -> None:
        actual = derive_swiglu_hidden_dim(dim=8, hidden_dim=None, multiple_of=8)

        self.assertEqual(actual, 24)


class SwiGLUTest(unittest.TestCase):
    def test_init_creates_expected_projection_shapes(self) -> None:
        module = SwiGLU(dim=8, hidden_dim=16, multiple_of=8, dropout=0.0)

        self.assertEqual(module.hidden_dim, 16)
        self.assertEqual(tuple(module.w1.weight.shape), (16, 8))
        self.assertEqual(tuple(module.w2.weight.shape), (8, 16))
        self.assertEqual(tuple(module.w3.weight.shape), (16, 8))

    def test_forward_returns_expected_shape(self) -> None:
        torch.manual_seed(0)
        module = SwiGLU(dim=8, hidden_dim=16, multiple_of=8, dropout=0.0)
        x = torch.randn(2, 3, 8)

        actual = module(x)

        self.assertEqual(tuple(actual.shape), (2, 3, 8))

    def test_forward_matches_reference_formula(self) -> None:
        torch.manual_seed(1)
        module = SwiGLU(dim=4, hidden_dim=6, multiple_of=2, dropout=0.0)
        module.eval()
        x = torch.randn(2, 3, 4)

        actual = module(x)
        expected = reference_swiglu(module, x)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_eval_mode_is_deterministic_when_dropout_is_configured(self) -> None:
        torch.manual_seed(2)
        module = SwiGLU(dim=4, hidden_dim=8, multiple_of=4, dropout=0.5)
        module.eval()
        x = torch.randn(1, 2, 4)

        first = module(x)
        second = module(x)

        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_visual_comparison_for_learning(self) -> None:
        torch.manual_seed(3)
        module = SwiGLU(dim=4, hidden_dim=6, multiple_of=2, dropout=0.0)
        module.eval()
        x = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [-1.0, 0.5, 0.0, 2.0],
                ]
            ]
        )

        actual = module(x)
        expected = reference_swiglu(module, x)
        diff = (actual - expected).abs()

        print("\n=== SwiGLU Visual Comparison ===")
        print(format_token_rows("input", x))
        print(format_token_rows("expected", expected))
        print(format_token_rows("actual", actual))
        print(format_token_rows("abs diff", diff))
        print(f"max_abs_diff: {diff.max().item():.8f}")

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

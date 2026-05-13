"""Unit tests and visual checks for scratch_llm.model.norm."""

from __future__ import annotations

import unittest

import torch

from scratch_llm.model.norm import RMSNorm


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm formula used by the tests."""

    x_float = x.float()
    rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_float / rms * weight).type_as(x)


def format_token_rows(name: str, tensor: torch.Tensor) -> str:
    """Format a (batch, seq, dim) tensor as easy-to-read token rows."""

    rows = [f"{name}:"]
    data = tensor.detach().cpu()
    for batch_idx in range(data.shape[0]):
        for token_idx in range(data.shape[1]):
            values = ", ".join(f"{v:8.4f}" for v in data[batch_idx, token_idx].tolist())
            rows.append(f"  batch={batch_idx}, token={token_idx}: [{values}]")
    return "\n".join(rows)


class RMSNormTest(unittest.TestCase):
    def test_rmsnorm_matches_reference_formula_per_token(self) -> None:
        norm = RMSNorm(dim=4, eps=1e-5)
        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, 0.0, 0.0]],
                [[-1.0, -2.0, -3.0, -4.0], [0.5, 0.5, 0.5, 0.5]],
            ]
        )

        actual = norm(x)
        expected = reference_rmsnorm(x, norm.weight, norm.eps)

        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_rmsnorm_uses_learnable_weight(self) -> None:
        norm = RMSNorm(dim=4, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        x = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        actual = norm(x)
        expected = reference_rmsnorm(x, norm.weight, norm.eps)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.allclose(actual[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-4))

    def test_rmsnorm_preserves_input_dtype(self) -> None:
        norm = RMSNorm(dim=4)
        x = torch.ones(2, 3, 4, dtype=torch.float16)

        actual = norm(x)

        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(actual.shape, x.shape)

    def test_visual_comparison_for_learning(self) -> None:
        norm = RMSNorm(dim=4, eps=1e-5)
        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, 0.0, 0.0]],
            ]
        )

        actual = norm(x)
        expected = reference_rmsnorm(x, norm.weight, norm.eps)
        diff = (actual - expected).abs()

        print("\n=== RMSNorm Visual Comparison ===")
        print(format_token_rows("input", x))
        print(format_token_rows("expected", expected))
        print(format_token_rows("actual", actual))
        print(format_token_rows("abs diff", diff))
        print(f"max_abs_diff: {diff.max().item():.8f}")

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

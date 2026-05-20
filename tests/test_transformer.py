"""Unit tests and visual checks for scratch_llm.model.transformer."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from scratch_llm.config import ModelConfig
from scratch_llm.model.blocks import DecoderBlock
from scratch_llm.model.norm import RMSNorm
from scratch_llm.model.transformer import ScratchLLM


def small_config(**overrides: object) -> ModelConfig:
    """Create a small model config suitable for CPU unit tests."""

    values = {
        "vocab_size": 17,
        "dim": 8,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": None,
        "hidden_dim": 16,
        "multiple_of": 8,
        "max_seq_len": 6,
        "dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_embeddings": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


def reference_forward(
    model: ScratchLLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference top-level Transformer forward path without loss."""

    seq_len = input_ids.shape[1]
    h = model.tok_embeddings(input_ids)
    h = model.dropout(h)
    freqs_cos = model.freqs_cos[:seq_len]
    freqs_sin = model.freqs_sin[:seq_len]
    normalized_attention_mask = model.prepare_attention_mask(attention_mask, input_ids)

    for layer in model.layers:
        h = layer(
            h,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            attention_mask=normalized_attention_mask,
        )

    h = model.norm(h)
    return model.output(h)


def reference_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Reference CE loss that ignores -100 and pad token labels."""

    labels = labels.clone()
    labels = labels.masked_fill(labels == pad_token_id, -100)
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def format_logits_rows(name: str, tensor: torch.Tensor, limit: int = 5) -> str:
    """Format a few logits from each token row for visual inspection."""

    rows = [f"{name}:"]
    data = tensor.detach().cpu()
    for batch_idx in range(data.shape[0]):
        for token_idx in range(data.shape[1]):
            values = ", ".join(
                f"{v:8.4f}" for v in data[batch_idx, token_idx, :limit].tolist()
            )
            rows.append(
                f"  batch={batch_idx}, token={token_idx}, first_{limit}: [{values}]"
            )
    return "\n".join(rows)


class ScratchLLMInitTest(unittest.TestCase):
    def test_init_creates_expected_modules_and_rope_buffers(self) -> None:
        config = small_config()
        model = ScratchLLM(config)

        self.assertEqual(tuple(model.tok_embeddings.weight.shape), (17, 8))
        self.assertEqual(len(model.layers), 2)
        self.assertIsInstance(model.layers[0], DecoderBlock)
        self.assertIsInstance(model.norm, RMSNorm)
        self.assertEqual(tuple(model.output.weight.shape), (17, 8))
        self.assertEqual(tuple(model.freqs_cos.shape), (6, 2))
        self.assertEqual(tuple(model.freqs_sin.shape), (6, 2))

    def test_tie_embeddings_shares_embedding_and_output_weight(self) -> None:
        tied_model = ScratchLLM(small_config(tie_embeddings=True))
        untied_model = ScratchLLM(small_config(tie_embeddings=False))

        self.assertIs(tied_model.output.weight, tied_model.tok_embeddings.weight)
        self.assertIsNot(untied_model.output.weight, untied_model.tok_embeddings.weight)

    def test_pad_embedding_is_initialized_to_zero(self) -> None:
        model = ScratchLLM(small_config(pad_token_id=0))

        self.assertTrue(torch.allclose(model.tok_embeddings.weight[0], torch.zeros(8)))


class AttentionMaskPreparationTest(unittest.TestCase):
    def test_none_mask_is_derived_from_pad_tokens(self) -> None:
        model = ScratchLLM(small_config())
        input_ids = torch.tensor([[1, 5, 0, 0], [1, 6, 7, 0]])

        actual = model.prepare_attention_mask(None, input_ids)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
            ]
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_2d_mask_is_converted_to_bool(self) -> None:
        model = ScratchLLM(small_config())
        input_ids = torch.ones(2, 3, dtype=torch.long)
        mask = torch.tensor([[1, 0, 1], [0, 1, 1]])

        actual = model.prepare_attention_mask(mask, input_ids)

        self.assertEqual(actual.dtype, torch.bool)
        self.assertTrue(torch.equal(actual, mask.bool()))

    def test_3d_and_4d_masks_are_normalized_to_key_mask(self) -> None:
        model = ScratchLLM(small_config())
        input_ids = torch.ones(1, 3, dtype=torch.long)
        mask_3d = torch.tensor(
            [
                [
                    [1, 0, 1],
                    [1, 0, 1],
                    [0, 0, 0],
                ]
            ]
        )
        mask_4d = mask_3d[:, None, :, :]

        expected = torch.tensor([[True, False, True]])

        self.assertTrue(torch.equal(model.prepare_attention_mask(mask_3d, input_ids), expected))
        self.assertTrue(torch.equal(model.prepare_attention_mask(mask_4d, input_ids), expected))

    def test_invalid_mask_shape_raises_value_error(self) -> None:
        model = ScratchLLM(small_config())
        input_ids = torch.ones(2, 3, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "3D attention_mask"):
            model.prepare_attention_mask(torch.ones(2, 2, 3), input_ids)

        with self.assertRaisesRegex(ValueError, "attention_mask must be"):
            model.prepare_attention_mask(torch.ones(2, 1, 1, 1, 3), input_ids)


class ScratchLLMForwardTest(unittest.TestCase):
    def test_forward_returns_logits_and_no_loss_without_labels(self) -> None:
        torch.manual_seed(0)
        model = ScratchLLM(small_config())
        model.eval()
        input_ids = torch.tensor([[1, 3, 4, 0], [1, 5, 6, 2]])

        output = model(input_ids)

        self.assertEqual(tuple(output["logits"].shape), (2, 4, 17))
        self.assertIsNone(output["loss"])

    def test_forward_logits_match_reference_path(self) -> None:
        torch.manual_seed(1)
        model = ScratchLLM(small_config())
        model.eval()
        input_ids = torch.tensor([[1, 3, 4, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        actual = model(input_ids, attention_mask=attention_mask)["logits"]
        expected = reference_forward(model, input_ids, attention_mask)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_forward_computes_cross_entropy_loss_and_ignores_pad_labels(self) -> None:
        torch.manual_seed(2)
        model = ScratchLLM(small_config())
        model.eval()
        input_ids = torch.tensor([[1, 3, 4, 0]])
        labels = torch.tensor([[3, 4, 0, -100]])

        output = model(input_ids, labels=labels)
        expected_loss = reference_loss(output["logits"], labels, pad_token_id=0)

        self.assertIsNotNone(output["loss"])
        self.assertTrue(torch.allclose(output["loss"], expected_loss, atol=1e-6))

    def test_forward_returns_zero_loss_when_all_labels_are_ignored(self) -> None:
        model = ScratchLLM(small_config())
        input_ids = torch.tensor([[1, 0, 0]])
        labels = torch.tensor([[0, -100, 0]])

        output = model(input_ids, labels=labels)

        self.assertTrue(torch.equal(output["loss"], torch.tensor(0.0)))

    def test_forward_rejects_invalid_input_shapes(self) -> None:
        model = ScratchLLM(small_config(max_seq_len=3))

        with self.assertRaisesRegex(ValueError, "input_ids must have shape"):
            model(torch.ones(1, 2, 3, dtype=torch.long))

        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            model(torch.ones(1, 4, dtype=torch.long))

        with self.assertRaisesRegex(ValueError, "labels must have shape"):
            model(torch.ones(1, 2, dtype=torch.long), labels=torch.ones(1, 3, dtype=torch.long))

    def test_visual_comparison_for_learning(self) -> None:
        torch.manual_seed(3)
        model = ScratchLLM(small_config(vocab_size=11, dim=4, n_heads=1, hidden_dim=8))
        model.eval()
        input_ids = torch.tensor([[1, 3, 4, 0]])
        labels = torch.tensor([[3, 4, 0, -100]])

        output = model(input_ids, labels=labels)
        expected_logits = reference_forward(model, input_ids)
        diff = (output["logits"] - expected_logits).abs()

        print("\n=== ScratchLLM Visual Comparison ===")
        print(f"input_ids: {input_ids.tolist()}")
        print(f"labels: {labels.tolist()}")
        print(format_logits_rows("expected logits", expected_logits))
        print(format_logits_rows("actual logits", output["logits"]))
        print(format_logits_rows("abs diff", diff))
        print(f"loss: {output['loss'].item():.6f}")
        print(f"max_abs_diff: {diff.max().item():.8f}")

        self.assertTrue(torch.allclose(output["logits"], expected_logits, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

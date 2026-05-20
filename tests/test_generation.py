"""Unit tests for scratch_llm.inference.generation."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from scratch_llm.config import GenerationConfig
from scratch_llm.inference.generation import generate, sample_next_token, top_k_filter


class ScheduledLogitModel(nn.Module):
    """Tiny causal model that emits scheduled next-token logits."""

    def __init__(self, scheduled_tokens: list[list[int]], vocab_size: int = 8) -> None:
        super().__init__()
        self.scheduled_tokens = scheduled_tokens
        self.vocab_size = vocab_size
        self.calls = 0
        self.seen_input_ids: list[torch.Tensor] = []
        self.seen_attention_masks: list[torch.Tensor | None] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        self.seen_input_ids.append(input_ids.detach().clone())
        if attention_mask is None:
            self.seen_attention_masks.append(None)
        else:
            self.seen_attention_masks.append(attention_mask.detach().clone())

        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            fill_value=-1000.0,
            device=input_ids.device,
        )

        step_tokens = self.scheduled_tokens[min(self.calls, len(self.scheduled_tokens) - 1)]
        for batch_idx, token_id in enumerate(step_tokens):
            logits[batch_idx, -1, token_id] = 1000.0

        self.calls += 1
        return {"logits": logits, "loss": None}


class TopKFilterTest(unittest.TestCase):
    def test_top_k_filter_keeps_only_highest_k_logits(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])

        actual = top_k_filter(logits, top_k=2)

        self.assertTrue(torch.isneginf(actual[0, 0]))
        self.assertEqual(actual[0, 1].item(), 5.0)
        self.assertTrue(torch.isneginf(actual[0, 2]))
        self.assertEqual(actual[0, 3].item(), 4.0)

    def test_top_k_filter_none_or_large_k_returns_original_logits(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        self.assertIs(top_k_filter(logits, None), logits)
        self.assertIs(top_k_filter(logits, 3), logits)
        self.assertIs(top_k_filter(logits, 10), logits)

    def test_top_k_filter_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "logits must have shape"):
            top_k_filter(torch.ones(2, 3, 4), top_k=2)

        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            top_k_filter(torch.ones(1, 4), top_k=0)


class SampleNextTokenTest(unittest.TestCase):
    def test_temperature_zero_uses_greedy_argmax(self) -> None:
        logits = torch.tensor([[0.1, 5.0, 4.0], [3.0, 2.0, 1.0]])

        actual = sample_next_token(logits, temperature=0.0)

        self.assertTrue(torch.equal(actual, torch.tensor([[1], [0]])))

    def test_sampling_with_top_k_one_is_deterministic(self) -> None:
        logits = torch.tensor([[0.1, 5.0, 4.0], [3.0, 2.0, 1.0]])

        actual = sample_next_token(logits, temperature=1.0, top_k=1)

        self.assertTrue(torch.equal(actual, torch.tensor([[1], [0]])))

    def test_sample_next_token_rejects_invalid_temperature_or_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            sample_next_token(torch.ones(1, 4), temperature=-1.0)

        with self.assertRaisesRegex(ValueError, "logits must have shape"):
            sample_next_token(torch.ones(1, 2, 4), temperature=1.0)


class GenerateTest(unittest.TestCase):
    def test_generate_greedy_tokens_excluding_prompt(self) -> None:
        model = ScheduledLogitModel([[2], [3], [4]], vocab_size=6)
        input_ids = torch.tensor([[1]])
        config = GenerationConfig(
            max_new_tokens=3,
            temperature=0.0,
            eos_token_id=None,
            pad_token_id=0,
        )

        actual = generate(model, input_ids, config)

        self.assertTrue(torch.equal(actual, torch.tensor([[2, 3, 4]])))
        self.assertEqual(model.calls, 3)

    def test_generate_stops_when_all_sequences_reach_eos(self) -> None:
        model = ScheduledLogitModel([[2, 4], [4, 3], [1, 1]], vocab_size=6)
        input_ids = torch.tensor([[1], [1]])
        config = GenerationConfig(
            max_new_tokens=5,
            temperature=0.0,
            eos_token_id=4,
            pad_token_id=0,
        )

        actual = generate(model, input_ids, config)

        self.assertTrue(torch.equal(actual, torch.tensor([[2, 4], [4, 0]])))
        self.assertEqual(model.calls, 2)

    def test_generate_extends_attention_mask(self) -> None:
        model = ScheduledLogitModel([[2], [3]], vocab_size=6)
        input_ids = torch.tensor([[1, 0]])
        attention_mask = torch.tensor([[1, 0]])
        config = GenerationConfig(
            max_new_tokens=2,
            temperature=0.0,
            eos_token_id=None,
            pad_token_id=0,
        )

        actual = generate(model, input_ids, config, attention_mask=attention_mask)

        self.assertTrue(torch.equal(actual, torch.tensor([[2, 3]])))
        self.assertTrue(torch.equal(model.seen_attention_masks[0], torch.tensor([[1, 0]])))
        self.assertTrue(torch.equal(model.seen_attention_masks[1], torch.tensor([[True, False, True]])))

    def test_generate_derives_attention_mask_from_pad_tokens_when_missing(self) -> None:
        model = ScheduledLogitModel([[2]], vocab_size=6)
        input_ids = torch.tensor([[1, 0]])
        config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            eos_token_id=None,
            pad_token_id=0,
        )

        generate(model, input_ids, config)

        self.assertTrue(torch.equal(model.seen_attention_masks[0], torch.tensor([[True, False]])))

    def test_generate_returns_empty_tensor_for_zero_new_tokens(self) -> None:
        model = ScheduledLogitModel([[2]], vocab_size=6)
        input_ids = torch.tensor([[1, 2]])
        config = GenerationConfig(max_new_tokens=0)

        actual = generate(model, input_ids, config)

        self.assertEqual(tuple(actual.shape), (1, 0))
        self.assertEqual(model.calls, 0)

    def test_generate_preserves_training_mode(self) -> None:
        model = ScheduledLogitModel([[2]], vocab_size=6)
        model.train()
        input_ids = torch.tensor([[1]])
        config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            eos_token_id=None,
            pad_token_id=0,
        )

        generate(model, input_ids, config)

        self.assertTrue(model.training)

    def test_generate_rejects_invalid_shapes(self) -> None:
        model = ScheduledLogitModel([[2]], vocab_size=6)
        config = GenerationConfig(max_new_tokens=1)

        with self.assertRaisesRegex(ValueError, "input_ids must have shape"):
            generate(model, torch.ones(1, 2, 3, dtype=torch.long), config)

        with self.assertRaisesRegex(ValueError, "attention_mask must have shape"):
            generate(
                model,
                torch.ones(1, 2, dtype=torch.long),
                config,
                attention_mask=torch.ones(1, 3),
            )

        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            generate(
                model,
                torch.ones(1, 2, dtype=torch.long),
                GenerationConfig(max_new_tokens=-1),
            )


if __name__ == "__main__":
    unittest.main()

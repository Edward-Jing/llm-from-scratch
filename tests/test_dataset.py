"""Unit tests for scratch_llm.data.dataset."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from scratch_llm.config import TokenizerConfig
from scratch_llm.data.dataset import (
    CausalLMDataset,
    SFTDataset,
    build_causal_lm_example,
    build_sft_loss_mask,
)
from scratch_llm.tokenizer import load_tokenizer, train_bpe_tokenizer


CASES = Path(__file__).parent / "dataset_cases"
TOKENIZER_CORPUS = Path(__file__).parent / "tokenizer_cases" / "tiny_corpus.jsonl"


def build_test_tokenizer(tmpdir: str):
    """Train and load a tiny tokenizer for dataset tests."""

    config = TokenizerConfig(vocab_size=160, min_frequency=1)
    train_bpe_tokenizer(TOKENIZER_CORPUS, tmpdir, config)
    return load_tokenizer(tmpdir)


class CausalLMExampleTest(unittest.TestCase):
    def test_build_causal_lm_example_pads_and_shifts(self) -> None:
        x, y, loss_mask = build_causal_lm_example(
            input_ids=[10, 11, 12],
            max_length=6,
            pad_token_id=0,
        )

        self.assertTrue(torch.equal(x, torch.tensor([10, 11, 12, 0, 0])))
        self.assertTrue(torch.equal(y, torch.tensor([11, 12, 0, 0, 0])))
        self.assertTrue(torch.equal(loss_mask, torch.tensor([1, 1, 0, 0, 0])))

    def test_build_causal_lm_example_truncates(self) -> None:
        x, y, loss_mask = build_causal_lm_example(
            input_ids=[1, 2, 3, 4, 5],
            max_length=4,
            pad_token_id=0,
        )

        self.assertTrue(torch.equal(x, torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(y, torch.tensor([2, 3, 4])))
        self.assertTrue(torch.equal(loss_mask, torch.tensor([1, 1, 1])))


class SFTLossMaskTest(unittest.TestCase):
    def test_build_sft_loss_mask_marks_only_assistant_span(self) -> None:
        mask = build_sft_loss_mask(
            input_ids=[1, 2, 3, 40, 41, 4, 5],
            assistant_prefix_ids=[1, 2, 3],
            eos_token_id=4,
        )

        self.assertEqual(mask, [0, 0, 0, 1, 1, 0, 0])

    def test_build_sft_loss_mask_rejects_empty_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_sft_loss_mask([1, 2, 3], [], eos_token_id=4)


class CausalLMDatasetTest(unittest.TestCase):
    def test_causal_lm_dataset_len_and_item_shape(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tokenizer = build_test_tokenizer(tmpdir)
            dataset = CausalLMDataset(
                CASES / "causal_pretrain.jsonl",
                tokenizer,
                max_length=12,
            )

            x, y, loss_mask = dataset[0]

        self.assertEqual(len(dataset), 3)
        self.assertEqual(tuple(x.shape), (11,))
        self.assertEqual(tuple(y.shape), (11,))
        self.assertEqual(tuple(loss_mask.shape), (11,))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(loss_mask.dtype, torch.long)
        self.assertGreater(int(loss_mask.sum().item()), 0)

    def test_causal_lm_dataset_supports_negative_index(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tokenizer = build_test_tokenizer(tmpdir)
            dataset = CausalLMDataset(
                CASES / "causal_pretrain.jsonl",
                tokenizer,
                max_length=10,
            )

            last_by_negative = dataset[-1]
            last_by_positive = dataset[len(dataset) - 1]

        self.assertTrue(torch.equal(last_by_negative[0], last_by_positive[0]))
        self.assertTrue(torch.equal(last_by_negative[1], last_by_positive[1]))
        self.assertTrue(torch.equal(last_by_negative[2], last_by_positive[2]))


class SFTDatasetTest(unittest.TestCase):
    def test_sft_dataset_len_shape_and_mask(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tokenizer = build_test_tokenizer(tmpdir)
            dataset = SFTDataset(
                CASES / "sft_messages.jsonl",
                tokenizer,
                max_length=48,
            )

            x, y, loss_mask = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(x.shape), (47,))
        self.assertEqual(tuple(y.shape), (47,))
        self.assertEqual(tuple(loss_mask.shape), (47,))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(loss_mask.dtype, torch.long)
        self.assertGreater(int(loss_mask.sum().item()), 0)
        self.assertLess(int(loss_mask.sum().item()), loss_mask.numel())

    def test_sft_dataset_accepts_raw_message_list_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tokenizer = build_test_tokenizer(tmpdir)
            dataset = SFTDataset(
                CASES / "sft_messages.jsonl",
                tokenizer,
                max_length=48,
            )

            x, y, loss_mask = dataset[1]

        self.assertEqual(tuple(x.shape), (47,))
        self.assertEqual(tuple(y.shape), (47,))
        self.assertEqual(tuple(loss_mask.shape), (47,))
        self.assertGreater(int(loss_mask.sum().item()), 0)

    def test_sft_dataset_rejects_non_list_messages(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tokenizer = build_test_tokenizer(tmpdir)
            dataset = SFTDataset(
                CASES / "sft_invalid_messages.jsonl",
                tokenizer,
                max_length=24,
            )

            with self.assertRaisesRegex(ValueError, "to be a list"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()

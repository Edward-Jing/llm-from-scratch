"""Unit tests for utility and inference CLI helpers."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from scratch_llm.inference.config_loader import build_model_config_from_checkpoint
from scratch_llm.utils import count_parameters, seed_everything


class CountParametersTest(unittest.TestCase):
    def test_count_parameters_can_include_or_exclude_frozen_weights(self) -> None:
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        for parameter in model[1].parameters():
            parameter.requires_grad = False

        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)

        self.assertEqual(trainable, 3 * 4 + 4)
        self.assertEqual(total, (3 * 4 + 4) + (4 * 2 + 2))


class SeedEverythingTest(unittest.TestCase):
    def test_seed_everything_reproducibly_seeds_python_and_torch(self) -> None:
        seed_everything(123)
        python_a = random.random()
        torch_a = torch.rand(3)

        seed_everything(123)
        python_b = random.random()
        torch_b = torch.rand(3)

        self.assertEqual(python_a, python_b)
        self.assertTrue(torch.equal(torch_a, torch_b))


class ConfigLoaderTest(unittest.TestCase):
    def test_build_model_config_reads_checkpoint_metadata_and_applies_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            torch.save(
                {
                    "model_state_dict": {},
                    "extra": {
                        "model_config": {
                            "vocab_size": 99,
                            "dim": 32,
                            "n_layers": 3,
                            "n_heads": 4,
                            "max_seq_len": 64,
                            "unknown_field": "ignored",
                        }
                    },
                },
                checkpoint_path,
            )

            config = build_model_config_from_checkpoint(
                checkpoint_path,
                overrides={"dim": 48, "n_heads": 6},
            )

        self.assertEqual(config.vocab_size, 99)
        self.assertEqual(config.dim, 48)
        self.assertEqual(config.n_layers, 3)
        self.assertEqual(config.n_heads, 6)
        self.assertEqual(config.max_seq_len, 64)


if __name__ == "__main__":
    unittest.main()

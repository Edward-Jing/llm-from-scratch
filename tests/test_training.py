"""Unit tests for scratch_llm.training.lr and scratch_llm.training.loop."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from scratch_llm.config import TrainConfig
from scratch_llm.training.loop import (
    compute_masked_loss,
    evaluate,
    move_batch_to_device,
    train_one_epoch,
)
from scratch_llm.training.lr import cosine_lr


class TinyLM(nn.Module):
    """Small language model used to test the training loop."""

    def __init__(self, vocab_size: int = 5, dim: int = 4, pad_token_id: int = 0) -> None:
        super().__init__()
        self.config = type("Config", (), {"pad_token_id": pad_token_id})()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden = self.embedding(input_ids)
        logits = self.output(hidden)
        return {"logits": logits, "loss": None}


class CosineLRTest(unittest.TestCase):
    def test_linear_warmup_then_cosine_decay(self) -> None:
        base_lr = 1.0

        self.assertAlmostEqual(cosine_lr(0, 10, base_lr, warmup_steps=2), 0.5)
        self.assertAlmostEqual(cosine_lr(1, 10, base_lr, warmup_steps=2), 1.0)

        mid = cosine_lr(6, 10, base_lr, warmup_steps=2, min_lr_ratio=0.1)
        expected_mid = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * 4 / 8))
        self.assertAlmostEqual(mid, expected_mid)

        self.assertAlmostEqual(cosine_lr(10, 10, base_lr, warmup_steps=2), 0.1)
        self.assertAlmostEqual(cosine_lr(100, 10, base_lr, warmup_steps=2), 0.1)

    def test_invalid_lr_arguments_raise_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_steps"):
            cosine_lr(0, 0, 1.0)
        with self.assertRaisesRegex(ValueError, "base_lr"):
            cosine_lr(0, 10, -1.0)
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            cosine_lr(0, 10, 1.0, warmup_steps=-1)
        with self.assertRaisesRegex(ValueError, "min_lr_ratio"):
            cosine_lr(0, 10, 1.0, min_lr_ratio=1.5)


class LoopUtilityTest(unittest.TestCase):
    def test_move_batch_to_device_moves_all_tensors(self) -> None:
        batch = (
            torch.tensor([[1, 2]]),
            torch.tensor([[2, 3]]),
            torch.tensor([[1, 0]]),
        )

        actual = move_batch_to_device(batch, "cpu")

        self.assertTrue(all(tensor.device.type == "cpu" for tensor in actual))

    def test_compute_masked_loss_averages_only_masked_positions(self) -> None:
        per_token_loss = torch.tensor([[1.0, 2.0, 10.0], [4.0, 8.0, 12.0]])
        loss_mask = torch.tensor([[1, 1, 0], [0, 1, 0]])

        actual = compute_masked_loss(per_token_loss, loss_mask)

        self.assertTrue(torch.allclose(actual, torch.tensor((1.0 + 2.0 + 8.0) / 3.0)))

    def test_compute_masked_loss_supports_flat_loss_shape(self) -> None:
        per_token_loss = torch.tensor([1.0, 2.0, 10.0, 4.0])
        loss_mask = torch.tensor([[1, 0], [1, 0]])

        actual = compute_masked_loss(per_token_loss, loss_mask)

        self.assertTrue(torch.allclose(actual, torch.tensor((1.0 + 10.0) / 2.0)))

    def test_compute_masked_loss_returns_zero_when_mask_is_empty(self) -> None:
        per_token_loss = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss_mask = torch.tensor([[0, 0]])

        actual = compute_masked_loss(per_token_loss, loss_mask)
        actual.backward()

        self.assertTrue(torch.equal(actual.detach(), torch.tensor(0.0)))
        self.assertTrue(torch.equal(per_token_loss.grad, torch.zeros_like(per_token_loss)))


class TrainLoopTest(unittest.TestCase):
    def make_config(self, **overrides: object) -> TrainConfig:
        values = {
            "device": "cpu",
            "dtype": "float32",
            "learning_rate": 0.01,
            "grad_clip": 1.0,
            "accumulation_steps": 1,
            "warmup_steps": 0,
            "log_interval": 1,
        }
        values.update(overrides)
        return TrainConfig(**values)

    def test_train_one_epoch_updates_parameters_and_logs(self) -> None:
        torch.manual_seed(0)
        model = TinyLM()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = self.make_config()
        dataloader = [
            (
                torch.tensor([[1, 2, 0]]),
                torch.tensor([[2, 3, 0]]),
                torch.tensor([[1, 1, 0]]),
            )
        ]
        logs: list[dict[str, object]] = []
        before = model.output.weight.detach().clone()

        step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            epoch=0,
            total_steps=1,
            logger=logs.append,
        )

        self.assertEqual(step, 1)
        self.assertFalse(torch.allclose(model.output.weight.detach(), before))
        self.assertEqual(len(logs), 1)
        self.assertIn("loss", logs[0])
        self.assertIn("lr", logs[0])

    def test_train_one_epoch_respects_gradient_accumulation(self) -> None:
        torch.manual_seed(1)
        model = TinyLM()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = self.make_config(accumulation_steps=2)
        dataloader = [
            (
                torch.tensor([[1, 2]]),
                torch.tensor([[2, 3]]),
                torch.tensor([[1, 1]]),
            ),
            (
                torch.tensor([[2, 3]]),
                torch.tensor([[3, 4]]),
                torch.tensor([[1, 1]]),
            ),
            (
                torch.tensor([[3, 4]]),
                torch.tensor([[4, 1]]),
                torch.tensor([[1, 1]]),
            ),
        ]

        step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            epoch=0,
            total_steps=2,
        )

        self.assertEqual(step, 2)

    def test_evaluate_returns_loss_and_perplexity(self) -> None:
        torch.manual_seed(2)
        model = TinyLM()
        config = self.make_config()
        dataloader = [
            (
                torch.tensor([[1, 2, 0]]),
                torch.tensor([[2, 3, 0]]),
                torch.tensor([[1, 1, 0]]),
            )
        ]

        metrics = evaluate(model, dataloader, config)

        self.assertGreater(metrics["loss"], 0.0)
        self.assertAlmostEqual(metrics["ppl"], math.exp(metrics["loss"]))

    def test_evaluate_respects_max_batches(self) -> None:
        torch.manual_seed(3)
        model = TinyLM()
        config = self.make_config()
        batch = (
            torch.tensor([[1, 2]]),
            torch.tensor([[2, 3]]),
            torch.tensor([[1, 1]]),
        )

        metrics = evaluate(model, [batch, batch], config, max_batches=1)

        self.assertGreater(metrics["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()

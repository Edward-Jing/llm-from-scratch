"""Unit tests for scratch_llm.training.checkpoint."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from scratch_llm.training.checkpoint import load_checkpoint, save_checkpoint, unwrap_model


class TinyNet(nn.Module):
    """Small module used for checkpoint round-trip tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModuleWrapper(nn.Module):
    """Minimal wrapper with a .module attribute like DataParallel/DDP."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


class CompileLikeWrapper(nn.Module):
    """Minimal wrapper with an ._orig_mod attribute like torch.compile."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._orig_mod = module


def train_one_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Populate model gradients and optimizer state."""

    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[0.5, -0.5]])
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


class UnwrapModelTest(unittest.TestCase):
    def test_unwrap_model_returns_plain_model(self) -> None:
        model = TinyNet()

        self.assertIs(unwrap_model(model), model)

    def test_unwrap_model_handles_module_wrappers(self) -> None:
        model = TinyNet()
        wrapped = ModuleWrapper(ModuleWrapper(model))

        self.assertIs(unwrap_model(wrapped), model)

    def test_unwrap_model_handles_compile_like_wrapper(self) -> None:
        model = TinyNet()
        wrapped = CompileLikeWrapper(model)

        self.assertIs(unwrap_model(wrapped), model)


class CheckpointRoundTripTest(unittest.TestCase):
    def test_save_checkpoint_creates_parent_directories_and_payload(self) -> None:
        model = TinyNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "model.pt"
            save_checkpoint(path, model, step=7, extra={"epoch": 2})

            self.assertTrue(path.exists())
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", payload)
            self.assertEqual(payload["step"], 7)
            self.assertEqual(payload["extra"], {"epoch": 2})

    def test_load_checkpoint_restores_model_weights_and_metadata(self) -> None:
        torch.manual_seed(0)
        source = TinyNet()
        target = TinyNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(path, source, step=11, extra={"note": "round-trip"})

            with torch.no_grad():
                for parameter in target.parameters():
                    parameter.zero_()

            metadata = load_checkpoint(path, target)

            for source_param, target_param in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.allclose(source_param, target_param))
            self.assertEqual(metadata["step"], 11)
            self.assertEqual(metadata["extra"], {"note": "round-trip"})

    def test_load_checkpoint_restores_optimizer_state(self) -> None:
        torch.manual_seed(1)
        source = TinyNet()
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
        train_one_step(source, source_optimizer)

        target = TinyNet()
        target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(path, source, optimizer=source_optimizer, step=3)
            load_checkpoint(path, target, optimizer=target_optimizer)

            source_state = source_optimizer.state_dict()
            target_state = target_optimizer.state_dict()
            self.assertEqual(len(target_state["state"]), len(source_state["state"]))
            self.assertEqual(
                target_state["param_groups"][0]["lr"],
                source_state["param_groups"][0]["lr"],
            )

    def test_load_checkpoint_accepts_legacy_model_key(self) -> None:
        model = TinyNet()
        target = TinyNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.pt"
            torch.save({"model": model.state_dict(), "step": 5}, path)

            metadata = load_checkpoint(path, target)

            self.assertEqual(metadata["step"], 5)
            for source_param, target_param in zip(model.parameters(), target.parameters()):
                self.assertTrue(torch.allclose(source_param, target_param))

    def test_load_checkpoint_reports_missing_and_unexpected_keys_when_not_strict(self) -> None:
        model = TinyNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "partial.pt"
            torch.save({"model_state_dict": {}, "step": 1}, path)

            metadata = load_checkpoint(path, model, strict=False)

            self.assertEqual(metadata["step"], 1)
            self.assertIn("linear.weight", metadata["missing_keys"])
            self.assertEqual(metadata["unexpected_keys"], [])

    def test_load_checkpoint_rejects_payload_without_model_state(self) -> None:
        model = TinyNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"step": 1}, path)

            with self.assertRaisesRegex(KeyError, "model state dict"):
                load_checkpoint(path, model)


if __name__ == "__main__":
    unittest.main()

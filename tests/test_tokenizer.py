"""Unit tests for scratch_llm.tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scratch_llm.config import TokenizerConfig
from scratch_llm.tokenizer import (
    DEFAULT_CHAT_TEMPLATE,
    load_tokenizer,
    special_tokens,
    train_bpe_tokenizer,
    validate_tokenizer,
    write_tokenizer_configs,
)


CASES = Path(__file__).parent / "tokenizer_cases"


class TokenizerConfigTest(unittest.TestCase):
    def test_special_tokens_are_in_deterministic_order(self) -> None:
        config = TokenizerConfig()

        self.assertEqual(
            special_tokens(config),
            ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"],
        )

    def test_write_tokenizer_configs_creates_expected_json_files(self) -> None:
        config = TokenizerConfig()

        with TemporaryDirectory() as tmpdir:
            write_tokenizer_configs(tmpdir, config)
            tokenizer_config = json.loads((Path(tmpdir) / "tokenizer_config.json").read_text())
            special_tokens_map = json.loads(
                (Path(tmpdir) / "special_tokens_map.json").read_text()
            )

        self.assertEqual(tokenizer_config["tokenizer_class"], "PreTrainedTokenizerFast")
        self.assertEqual(tokenizer_config["chat_template"], DEFAULT_CHAT_TEMPLATE)
        self.assertEqual(special_tokens_map["bos_token"], "<|im_start|>")
        self.assertEqual(special_tokens_map["eos_token"], "<|im_end|>")
        self.assertEqual(special_tokens_map["additional_special_tokens"], ["<s>", "</s>"])


class TokenizerTrainingTest(unittest.TestCase):
    def test_train_load_and_validate_tokenizer(self) -> None:
        config = TokenizerConfig(vocab_size=128, min_frequency=1)

        with TemporaryDirectory() as tmpdir:
            train_bpe_tokenizer(CASES / "tiny_corpus.jsonl", tmpdir, config)

            output_dir = Path(tmpdir)
            self.assertTrue((output_dir / "tokenizer.json").exists())
            self.assertTrue((output_dir / "tokenizer_config.json").exists())
            self.assertTrue((output_dir / "special_tokens_map.json").exists())

            tokenizer = load_tokenizer(output_dir)
            report = validate_tokenizer(tokenizer, "hello tokenizer")

        self.assertGreaterEqual(report["vocab_size"], 5)
        self.assertIn("<|im_start|>", report["special_tokens_map"].values())
        self.assertEqual(report["sample_text"], "hello tokenizer")
        self.assertIsInstance(report["encoded_ids"], list)
        self.assertIsInstance(report["decoded_text"], str)

    def test_train_tokenizer_respects_custom_text_key(self) -> None:
        config = TokenizerConfig(vocab_size=96, min_frequency=1, text_key="content")

        with TemporaryDirectory() as tmpdir:
            train_bpe_tokenizer(CASES / "custom_text_key.jsonl", tmpdir, config)
            tokenizer = load_tokenizer(tmpdir)
            ids = tokenizer.encode("custom key corpus")

        self.assertGreater(len(ids), 0)

    def test_validate_tokenizer_includes_chat_template_output(self) -> None:
        config = TokenizerConfig(vocab_size=128, min_frequency=1)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        with TemporaryDirectory() as tmpdir:
            train_bpe_tokenizer(CASES / "tiny_corpus.jsonl", tmpdir, config)
            tokenizer = load_tokenizer(tmpdir)
            report = validate_tokenizer(tokenizer, "hello", messages=messages)

        self.assertIn("<|im_start|>system", report["chat_text"])
        self.assertIn("<|im_start|>user", report["chat_text"])
        self.assertTrue(report["chat_text"].endswith("<|im_start|>assistant\n"))
        self.assertIsInstance(report["chat_encoded_ids"], list)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for scratch_llm.data.jsonl."""

from __future__ import annotations

from pathlib import Path
import unittest

from scratch_llm.data.jsonl import count_jsonl_lines, iter_jsonl_records, iter_jsonl_texts


CASES = Path(__file__).parent / "jsonl_cases"


class JsonlReaderTest(unittest.TestCase):
    def test_iter_jsonl_records_returns_dicts(self) -> None:
        records = list(iter_jsonl_records(CASES / "valid.jsonl"))

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["text"], "hello world")
        self.assertEqual(records[2]["text"], "中文也应该正常读取")

    def test_blank_lines_are_skipped(self) -> None:
        texts = list(iter_jsonl_texts(CASES / "with_blank_lines.jsonl"))

        self.assertEqual(texts, ["first", "second", "third"])
        self.assertEqual(count_jsonl_lines(CASES / "with_blank_lines.jsonl"), 3)

    def test_invalid_json_raises_value_error_with_line_number(self) -> None:
        with self.assertRaisesRegex(ValueError, "line 2"):
            list(iter_jsonl_records(CASES / "invalid_json.jsonl"))

    def test_missing_text_key_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing text"):
            list(iter_jsonl_texts(CASES / "missing_text.jsonl"))

    def test_non_string_text_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected text to be a string"):
            list(iter_jsonl_texts(CASES / "non_string_text.jsonl"))

    def test_non_object_record_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object"):
            list(iter_jsonl_records(CASES / "non_object_record.jsonl"))


if __name__ == "__main__":
    unittest.main()

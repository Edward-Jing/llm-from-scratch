"""JSONL reading contracts for pretraining and SFT data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Generator


def iter_jsonl_records(path: str | Path, encoding: str = "utf-8") -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a JSONL file.

    Args:
        path: File path to a .jsonl dataset.
        encoding: Text encoding used by the file.

    Returns:
        An iterator of dictionaries, one per line.

    TODO:
        Open the file, parse each non-empty line with json.loads, and include a
        useful line number in errors.
    """
    with Path(path).open(encoding=encoding) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line == "" :
                continue
            try:
                record = json.loads(line)
            except json.decoder.JSONDecodeError as error:
                raise ValueError(f"Failed to decode JSONL on line {line_number} of {path}") from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} of {path}, "
                    f"got {type(record).__name__}"
                )
            yield record


def iter_jsonl_texts(
    path: str | Path,
    text_key: str = "text",
    encoding: str = "utf-8",
) -> Generator[str, None, ValueError | None]:
    """Yield text fields from JSONL records.

    Args:
        path: File path to a .jsonl dataset.
        text_key: Key containing the raw text for tokenizer/pretraining data.
        encoding: Text encoding used by the file.

    Returns:
        An iterator of text strings.
    """
    for record_number, record in enumerate(iter_jsonl_records(path,encoding=encoding), start=1):
        if text_key not in record:
            raise ValueError(f"Missing {text_key} in record{record_number} of {path}")
        text = record[text_key]
        if not isinstance(text, str):
            raise ValueError(
                f"Expected {text_key} to be a string, "
                f"got {type(text)}"
                f"in record{record_number} of {path}")

        yield text
def count_jsonl_lines(path: str | Path) -> int:
    """Count dataset rows without loading the whole file into memory.

    Args:
        path: File path to a .jsonl dataset.
        encoding: Text encoding used by the file.

    Returns:
        Number of newline-delimited records.
    """
    count = 0
    with Path(path).open() as file:
        for line in file:
            if line.strip() != "":
                count += 1
    return count


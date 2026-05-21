"""Build a small SFT dataset from the public-domain pretraining JSONL.

The goal is not to create a strong assistant dataset. It creates lightweight
instruction-response examples that teach the chat template and assistant-only
loss path using copyright-clear project data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


def iter_records(path: str | Path) -> Iterable[dict[str, object]]:
    """Yield JSON object rows from a JSONL file."""

    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number}")
            yield record


def clean_excerpt(text: str, max_chars: int) -> str:
    """Return a compact excerpt suitable for an instruction prompt."""

    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def split_continuation(text: str, prompt_chars: int, answer_chars: int) -> tuple[str, str] | None:
    """Split one text chunk into a short prefix and continuation."""

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < prompt_chars + 80:
        return None

    prompt = text[:prompt_chars].rsplit(" ", 1)[0].strip()
    rest = text[len(prompt) :].strip()
    answer = rest[:answer_chars].rsplit(" ", 1)[0].strip()
    if not prompt or not answer:
        return None
    return prompt, answer


def make_messages(user: str, assistant: str) -> dict[str, list[dict[str, str]]]:
    """Create one chat-formatted JSONL row."""

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a small language model trained from scratch. "
                    "Answer clearly and stay close to the provided context."
                ),
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_examples(records: Iterable[dict[str, object]], max_examples: int) -> list[dict[str, object]]:
    """Build SFT examples from source metadata and text chunks."""

    examples: list[dict[str, object]] = []
    seen_sources: set[tuple[str, str]] = set()

    for record in records:
        text = str(record.get("text", "")).strip()
        source = str(record.get("source", "Unknown source"))
        author = str(record.get("author", "Unknown author"))
        if not text:
            continue

        source_key = (source, author)
        if source_key not in seen_sources:
            examples.append(
                make_messages(
                    user=f"What text is {source} from, and who wrote it?",
                    assistant=f"{source} is a public-domain text by {author}.",
                )
            )
            seen_sources.add(source_key)

        excerpt = clean_excerpt(text, max_chars=360)
        if excerpt:
            examples.append(
                make_messages(
                    user=(
                        "Identify the source of this excerpt and answer in one sentence:\n\n"
                        f"{excerpt}"
                    ),
                    assistant=f"This excerpt is from {source} by {author}.",
                )
            )

        split = split_continuation(text, prompt_chars=220, answer_chars=260)
        if split is not None:
            prompt, continuation = split
            examples.append(
                make_messages(
                    user=(
                        "Continue this passage in the same style, using only a short continuation:\n\n"
                        f"{prompt}"
                    ),
                    assistant=continuation,
                )
            )

        if len(examples) >= max_examples:
            break

    return examples[:max_examples]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build SFT JSONL from public-domain corpus")
    parser.add_argument("--input", default="data/pretrain_public_domain.jsonl")
    parser.add_argument("--output", default="data/sft_public_domain_chat.jsonl")
    parser.add_argument("--max-examples", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    """Create the SFT dataset."""

    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_examples(iter_records(args.input), max_examples=args.max_examples)
    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"wrote_examples={len(examples)} output={output_path}")


if __name__ == "__main__":
    main()

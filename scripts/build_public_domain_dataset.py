"""Build a small public-domain JSONL corpus for scratch pretraining.

This script downloads plain-text books from Project Gutenberg, strips the
Gutenberg header/footer, chunks the book bodies, and writes JSONL rows with a
single ``text`` field consumed by ``CausalLMDataset``.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    """Public-domain text source metadata."""

    title: str
    author: str
    ebook_id: int
    url: str


SOURCES = [
    Source(
        title="Alice's Adventures in Wonderland",
        author="Lewis Carroll",
        ebook_id=11,
        url="https://www.gutenberg.org/ebooks/11.txt.utf-8",
    ),
    Source(
        title="The Time Machine",
        author="H. G. Wells",
        ebook_id=35,
        url="https://www.gutenberg.org/ebooks/35.txt.utf-8",
    ),
    Source(
        title="The Strange Case of Dr. Jekyll and Mr. Hyde",
        author="Robert Louis Stevenson",
        ebook_id=43,
        url="https://www.gutenberg.org/ebooks/43.txt.utf-8",
    ),
    Source(
        title="The Wonderful Wizard of Oz",
        author="L. Frank Baum",
        ebook_id=55,
        url="https://www.gutenberg.org/ebooks/55.txt.utf-8",
    ),
    Source(
        title="The Adventures of Sherlock Holmes",
        author="Arthur Conan Doyle",
        ebook_id=1661,
        url="https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    ),
    Source(
        title="Pride and Prejudice",
        author="Jane Austen",
        ebook_id=1342,
        url="https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    ),
]


def download_text(url: str, timeout: int = 30) -> str:
    """Download UTF-8 text from a URL."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-from-scratch-dataset-builder/0.1"},
    )
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer around the book body."""

    start_match = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text)
    end_match = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text)

    if start_match is not None:
        text = text[start_match.end() :]
    if end_match is not None:
        text = text[: end_match.start()]
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int) -> list[str]:
    """Split text into paragraph-aware chunks."""

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if current else 0)
        if current and current_len + addition > chunk_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, len(paragraph), chunk_chars):
                piece = paragraph[start : start + chunk_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        current.append(paragraph)
        current_len += addition

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def write_sources_file(path: Path) -> None:
    """Write a short source manifest for the generated dataset."""

    lines = [
        "# Public-Domain Training Data Sources",
        "",
        "This dataset is built from public-domain texts hosted by Project Gutenberg.",
        "Each listed eBook page states `Copyright Public domain in the USA.`",
        "The generated JSONL file strips the Project Gutenberg header/footer and",
        "uses only the book-body text for small-scale local pretraining.",
        "",
        "| Title | Author | eBook | Source |",
        "| --- | --- | ---: | --- |",
    ]
    for source in SOURCES:
        page_url = f"https://www.gutenberg.org/ebooks/{source.ebook_id}"
        lines.append(
            f"| {source.title} | {source.author} | {source.ebook_id} | {page_url} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build small public-domain pretraining JSONL")
    parser.add_argument("--output", default="data/pretrain_public_domain.jsonl")
    parser.add_argument("--sources-output", default="data/SOURCES.md")
    parser.add_argument("--chunk-chars", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=2400)
    return parser.parse_args()


def main() -> None:
    """Download sources, chunk text, and write JSONL."""

    args = parse_args()
    output_path = Path(args.output)
    sources_path = Path(args.sources_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source in SOURCES:
        raw_text = download_text(source.url)
        body = normalize_text(strip_gutenberg_boilerplate(raw_text))
        for chunk_index, chunk in enumerate(chunk_text(body, args.chunk_chars)):
            rows.append(
                {
                    "text": chunk,
                    "source": source.title,
                    "author": source.author,
                    "ebook_id": source.ebook_id,
                    "chunk_index": chunk_index,
                }
            )

    rows = rows[: args.max_records]
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_sources_file(sources_path)
    total_chars = sum(len(str(row["text"])) for row in rows)
    print(f"wrote_records={len(rows)} output={output_path} total_chars={total_chars}")
    print(f"wrote_sources={sources_path}")


if __name__ == "__main__":
    main()

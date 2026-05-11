"""Tokenizer training and validation contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from scratch_llm.config import TokenizerConfig


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}" 
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def special_tokens(config: TokenizerConfig) -> list[str]:
    """Return special tokens in deterministic ID order.

    Args:
        config: TokenizerConfig containing token strings.

    Returns:
        List ordered as unk, optional legacy BOS/EOS, chat BOS, chat EOS.
        unk_token: Unknown token string.
        optional legacy BOS: historical beginning of sequence <s>
        optional legacy EOS: historical end of sequence  </s>
        bos_token: Beginning-of-sequence token string.
        eos_token: End-of-sequence token string.
    """

    return [config.unk_token, "<s>", "</s>", config.bos_token, config.eos_token]


def train_bpe_tokenizer(
    data_path: str | Path,
    output_dir: str | Path,
    config: TokenizerConfig,
) -> None:
    """Train and save a ByteLevel BPE tokenizer.

    Args:
        data_path: JSONL file with raw text.
        output_dir: Directory where tokenizer.json and configs are written.
        config: Tokenizer training config.

    TODO:
        Use tokenizers.Tokenizer(models.BPE), NFKC normalization, ByteLevel
        pre-tokenization/decoding, and BpeTrainer.
    """
    from tokenizers import Tokenizer,models,normalizers,pre_tokenizers,decoders, trainers
    from scratch_llm.data.jsonl import iter_jsonl_texts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Set trainer
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=special_tokens(config),
    )

    # Train tokenizer
    tokenizer.train_from_iterator(
        iter_jsonl_texts(data_path, text_key= config.text_key),
        trainer=trainer,
    )
    tokenizer.save(str(output_path / "tokenizer.json"))
    write_tokenizer_configs(output_path, config)

def write_tokenizer_configs(
    output_dir: str | Path,
    config: TokenizerConfig,
    chat_template: str = DEFAULT_CHAT_TEMPLATE,
) -> None:
    """Write tokenizer_config.json and special_tokens_map.json.

    Args:
        output_dir: Directory that already contains tokenizer.json.
        config: TokenizerConfig containing token strings.
        chat_template: Jinja chat template for transformers tokenizers.
    """
    import json
    output_path = Path(output_dir)
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": config.unk_token,
        "bos_token": config.bos_token,
        "eos_token": config.eos_token,
        "pad_token": config.pad_token,
        "chat_template": chat_template,
    }

    special_tokens_map = {
        "unk_token": config.unk_token,
        "bos_token": config.bos_token,
        "eos_token": config.eos_token,
        "pad_token": config.pad_token,
        "additional_special_tokens": ["<s>", "</s>"],
    }

    (output_path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_path / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_tokenizer(tokenizer_dir: str | Path) -> Any:
    """Load a saved tokenizer through transformers.AutoTokenizer.

    Args:
        tokenizer_dir: Directory containing tokenizer.json and config files.

    Returns:
        A tokenizer object compatible with the dataset and scripts.
    """
    from transformers import AutoTokenizer
    tokenizer_path = Path(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def validate_tokenizer(
    tokenizer: Any,
    sample_text: str,
    messages: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """Run simple encode/decode checks.

    Args:
        tokenizer: Tokenizer returned by load_tokenizer.
        sample_text: Text used for round-trip encoding checks.
        messages: Optional chat messages for chat_template checks.

    Returns:
        Dictionary with vocab size, special tokens, encoded IDs, and decoded text.
    """
    encoded_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_ids)

    result = {
        "vocab_size": len(tokenizer),
        "special_tokens_map": tokenizer.special_tokens_map,
        "sample_text": sample_text,
        "encoded_ids": encoded_ids,
        "decoded_text": decoded_text,
    }

    if messages is not None:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        chat_encoded_ids = tokenizer.encode(chat_text)
        result["chat_text"] = chat_text
        result["chat_encoded_ids"] = chat_encoded_ids

    return result


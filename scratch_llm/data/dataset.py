"""Dataset contracts for next-token prediction."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import Dataset
import json


def build_causal_lm_example(
    input_ids: list[int],
    max_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one shifted causal-LM training example.

    Args:
        input_ids: Token IDs before shifting. Include BOS/EOS if desired.
        max_length: Final unshifted sequence length before creating X and Y.
        pad_token_id: Token ID used to pad short examples.

    Returns:
        A tuple (x, y, loss_mask):
        x: Input IDs with shape (max_length - 1,).
        y: Target IDs shifted one token left, shape (max_length - 1,).
        loss_mask: 1 for real target positions and 0 for padding targets.
    """
    # truncate input_ids using max_length
    ids = input_ids[0:max_length]
    real_id_length = len(ids)

    if real_id_length < max_length:
        ids = ids + [pad_token_id] * (max_length - real_id_length)

    x_ids = ids[:-1]
    y_ids = ids[1:]
    loss_mask = [1] * max(real_id_length-1,0) + [0] * (max_length - real_id_length)

    return(
        torch.tensor(x_ids, dtype=torch.long),
        torch.tensor(y_ids,dtype=torch.long),
        torch.tensor(loss_mask,dtype=torch.long),
    )


def build_sft_loss_mask(
    input_ids: list[int],
    assistant_prefix_ids: list[int],
    eos_token_id: int,
) -> list[int]:
    """Mark only assistant answer spans for supervised fine-tuning loss.

    Args:
        input_ids: Full chat prompt token IDs.
        assistant_prefix_ids: Token IDs that identify '<assistant> starts here'.
        eos_token_id: Token ID that closes an assistant answer.

    Returns:
        A list with the same length as input_ids where assistant answer tokens
        are 1 and all prompt/system/user tokens are 0.
    """
    if not assistant_prefix_ids:
        raise ValueError("assistant_prefix_ids must not be empty")

    mask = [0] * len(input_ids)
    prefix_length = len(assistant_prefix_ids)
    for i in range(0, len(input_ids)):
        if input_ids[i:i+ prefix_length] == assistant_prefix_ids:
            j = i + prefix_length
            while j < len(input_ids) and input_ids[j] != eos_token_id:
                mask[j] = 1
                j += 1
    return mask




class CausalLMDataset(Dataset):
    """JSONL pretraining dataset for next-token prediction.

    Args:
        data_path: JSONL file path. Each row should contain a text field.
        tokenizer: Tokenizer object with __call__, bos_token, and pad_token_id.
        max_length: Final sequence length before shifting.
        text_key: JSON key containing raw text.
        bos_token: Optional token string to prepend. Defaults to tokenizer.bos_token.
        pad_token_id: Optional padding token ID. Defaults to tokenizer.pad_token_id.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        max_length: int,
        text_key: str = "text",
        bos_token: Optional[str] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key

        self.bos_token = tokenizer.bos_token if bos_token is None else bos_token
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id

        if self.pad_token_id is None:
            self.pad_token_id = 0

        self._offsets: list[int] = []

        with self.data_path.open("rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()

                if not line:
                    break

                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        """Return the number of JSONL rows."""
        return len(self._offsets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one row, tokenize it, and return (x, y, loss_mask).

        Args:
            index: Integer row index.

        Returns:
            A training tuple from build_causal_lm_example.
        """
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        with self.data_path.open("rb") as file:
            file.seek(self._offsets[index])
            line = file.readline().decode("utf-8")

        record = json.loads(line)

        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at index {index}, got {type(record).__name__}")

        if self.text_key not in record:
            raise ValueError(f"Missing {self.text_key} in record {index}")

        text = record[self.text_key]

        if not isinstance(text, str):
            raise ValueError(
                f"Expected {self.text_key} to be a string, "
                f"got {type(text).__name__} in record {index}"
            )

        if self.bos_token is not None:
            text = self.bos_token + text

        encoded = self.tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]

        return build_causal_lm_example(
            input_ids=input_ids,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
        )


class SFTDataset(Dataset):
    """JSONL chat dataset for instruction tuning.

    Args:
        data_path: JSONL file path. Each row should be a list of chat messages
            or contain a messages field.
        tokenizer: Tokenizer object with apply_chat_template and __call__.
        max_length: Final sequence length before shifting.
        messages_key: JSON key containing messages when rows are dictionaries.
        assistant_prefix: String prefix used to locate assistant spans.
        pad_token_id: Optional padding token ID. Defaults to tokenizer.pad_token_id.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        max_length: int,
        messages_key: str = "messages",
        assistant_prefix: str = "<|im_start|>assistant\n",
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages_key = messages_key
        self.assistant_prefix = assistant_prefix
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id

        if self.pad_token_id is None:
            self.pad_token_id = 0

        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id must not be None for SFTDataset")
        self.eos_token_id = tokenizer.eos_token_id

        self.assistant_prefix_ids = tokenizer(
            assistant_prefix,
            add_special_tokens=False,
        )["input_ids"]

        self._offsets: list[int] = []

        with self.data_path.open("rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()

                if not line:
                    break

                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        """Return the number of JSONL rows."""

        return len(self._offsets)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one chat row and return shifted tensors with SFT loss mask.

        Args:
            index: Integer row index.
        Returns:
            A tuple (x, y, loss_mask), each with shape (max_length - 1,).
        """
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        with self.data_path.open("rb") as file:
            file.seek(self._offsets[index])
            line = file.readline().decode("utf-8")

        record = json.loads(line)

        if isinstance(record, list):
            messages = record
        elif isinstance(record, dict):
            if self.messages_key not in record:
                raise ValueError(f"Missing {self.messages_key} in record {index}")
            messages = record[self.messages_key]
        else:
            raise ValueError(
                f"Expected JSON object or list at index {index}, "
                f"got {type(record).__name__}"
            )

        if not isinstance(messages, list):
            raise ValueError(
                f"Expected {self.messages_key} to be a list, "
                f"got {type(messages).__name__} in record {index}"
            )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = input_ids[: self.max_length]
        real_length = len(input_ids)

        loss_mask = build_sft_loss_mask(
            input_ids=input_ids,
            assistant_prefix_ids=self.assistant_prefix_ids,
            eos_token_id=self.eos_token_id,
        )

        if real_length < self.max_length:
            pad_length = self.max_length - real_length
            input_ids = input_ids + [self.pad_token_id] * pad_length
            loss_mask = loss_mask + [0] * pad_length

        x_ids = input_ids[:-1]
        y_ids = input_ids[1:]
        loss_mask = loss_mask[1:]

        return (
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(y_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.long),
        )
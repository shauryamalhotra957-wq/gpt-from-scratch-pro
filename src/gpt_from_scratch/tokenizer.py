from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer protocol used by training and generation."""

    @property
    def vocab_size(self) -> int:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: Iterable[int]) -> str:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...

    def save(self, path: str | Path) -> None:
        ...


@dataclass
class CharacterTokenizer:
    """Small, transparent character tokenizer for Karpathy-style experiments."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def train(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text")
        chars = sorted(set(text))
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            char = exc.args[0]
            raise ValueError(
                f"Character {char!r} was not in the tokenizer vocabulary"
            ) from exc

    def decode(self, ids: Iterable[int]) -> str:
        chars: list[str] = []
        for token_id in ids:
            try:
                chars.append(self.itos[int(token_id)])
            except KeyError as exc:
                raise ValueError(f"Token id {token_id!r} is not in the vocabulary") from exc
        return "".join(chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "char",
            "stoi": self.stoi,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharacterTokenizer":
        stoi = {str(key): int(value) for key, value in payload["stoi"].items()}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def save(self, path: str | Path) -> None:
        _save_tokenizer(path, self.to_dict())


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


@dataclass
class BytePairTokenizer:
    """A compact byte-level BPE tokenizer implemented from first principles.

    Token ids 0-255 represent raw bytes. Learned merges start at id 256, which
    makes the tokenizer reversible for arbitrary UTF-8 text without needing an
    unknown token.
    """

    merges: list[tuple[int, int, int]]

    def __post_init__(self) -> None:
        known_ids = set(range(256))
        for left, right, new_id in self.merges:
            if new_id < 256 or new_id in known_ids:
                raise ValueError(f"Invalid BPE token id: {new_id}")
            if left not in known_ids or right not in known_ids:
                raise ValueError(
                    f"BPE merge references unknown token(s): {left}, {right}"
                )
            known_ids.add(new_id)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "BytePairTokenizer":
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text")
        if vocab_size < 256:
            raise ValueError("byte-level BPE vocab_size must be at least 256")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int, int]] = []
        next_id = 256

        while next_id < vocab_size and len(ids) >= 2:
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break
            (left, right), frequency = pair_counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            ids = _merge_pair(ids, (left, right), next_id)
            merges.append((left, right, next_id))
            next_id += 1

        return cls(merges=merges)

    @property
    def vocab_size(self) -> int:
        if not self.merges:
            return 256
        return max(new_id for _, _, new_id in self.merges) + 1

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for left, right, new_id in self.merges:
            ids = _merge_pair(ids, (left, right), new_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        vocab = self._expanded_vocab()
        output = bytearray()
        for token_id in ids:
            token = int(token_id)
            if token not in vocab:
                raise ValueError(f"Token id {token!r} is not in the vocabulary")
            output.extend(vocab[token])
        return bytes(output).decode("utf-8", errors="replace")

    def _expanded_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for left, right, new_id in self.merges:
            vocab[new_id] = vocab[left] + vocab[right]
        return vocab

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "byte_bpe",
            "merges": [list(merge) for merge in self.merges],
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BytePairTokenizer":
        merges = [tuple(int(value) for value in merge) for merge in payload["merges"]]
        return cls(merges=[(left, right, new_id) for left, right, new_id in merges])

    def save(self, path: str | Path) -> None:
        _save_tokenizer(path, self.to_dict())


def _save_tokenizer(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def tokenizer_from_dict(payload: dict[str, Any]) -> Tokenizer:
    tokenizer_type = payload.get("type")
    if tokenizer_type == "char":
        return CharacterTokenizer.from_dict(payload)
    if tokenizer_type == "byte_bpe":
        return BytePairTokenizer.from_dict(payload)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type!r}")


def tokenizer_from_file(path: str | Path) -> Tokenizer:
    with Path(path).open("r", encoding="utf-8") as handle:
        return tokenizer_from_dict(json.load(handle))


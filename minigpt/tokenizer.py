from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """Small, deterministic character tokenizer for a self-contained baseline.

    This is intentionally simple, not a production tokenizer. Fit on the training
    corpus only, then reuse the same vocabulary for validation and inference.
    """
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = sorted(set(text))
        return cls({ch: i for i, ch in enumerate(vocab)}, vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"Text contains characters not in vocabulary: {unknown[:8]!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        if any(i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("Token ID outside vocabulary.")
        return "".join(self.itos[i] for i in ids)

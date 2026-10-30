from pathlib import Path
import torch


def load_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if len(text.strip()) < 2:
        raise ValueError("Training text must contain at least two non-whitespace characters.")
    return text


def split_ids(ids: list[int], train_fraction: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.5 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0.5 and 1.0")
    if len(ids) < 4:
        raise ValueError("Need at least four token IDs to create train and validation splits.")
    cutoff = int(len(ids) * train_fraction)
    return torch.tensor(ids[:cutoff], dtype=torch.long), torch.tensor(ids[cutoff:], dtype=torch.long)


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1 or block_size < 1:
        raise ValueError("batch_size and block_size must be positive")
    if data.numel() <= block_size:
        raise ValueError("Dataset must contain more tokens than block_size")
    starts = torch.randint(0, data.numel() - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

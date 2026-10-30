import pytest
import torch
from minigpt.data import get_batch, split_ids


def test_split_is_ordered_and_nonempty():
    train, val = split_ids(list(range(100)))
    assert train.numel() == 90
    assert val.numel() == 10
    assert train[-1].item() < val[0].item()


def test_batch_next_token_alignment():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8, device=torch.device("cpu"))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:]) or (y - x).eq(1).all()


def test_short_data_rejected():
    with pytest.raises(ValueError):
        get_batch(torch.arange(4), 2, 4, torch.device("cpu"))

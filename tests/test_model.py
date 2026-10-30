import pytest
import torch
from minigpt.model import MiniGPT


def tiny_model():
    return MiniGPT(vocab_size=17, block_size=12, n_embd=16, n_head=4, n_layer=2, dropout=0.0)


def test_forward_shape_and_loss():
    model = tiny_model()
    x = torch.randint(0, 17, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 17)
    assert loss is not None and torch.isfinite(loss)


def test_sequence_over_context_raises():
    model = tiny_model()
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 17, (1, 13)))


def test_generate_length():
    model = tiny_model()
    prompt = torch.randint(0, 17, (1, 5))
    result = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=5)
    assert result.shape == (1, 9)


def test_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        tiny_model().generate(torch.ones((1, 2), dtype=torch.long), 2, temperature=0)

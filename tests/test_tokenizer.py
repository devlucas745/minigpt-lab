import pytest
from minigpt.tokenizer import CharTokenizer


def test_round_trip():
    tokenizer = CharTokenizer.from_text("hello world")
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


def test_unknown_character_is_explicit():
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in vocabulary"):
        tokenizer.encode("z")


def test_vocab_is_deterministic():
    assert CharTokenizer.from_text("cab").itos == CharTokenizer.from_text("abc").itos

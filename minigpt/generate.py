import argparse
import torch

from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer


def load_checkpoint(path: str, device: torch.device):
    # Only load checkpoints you created or otherwise trust.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(**checkpoint["tokenizer"])
    model = MiniGPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/minigpt.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    # Unknown prompt characters are reported clearly by the tokenizer.
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(ids, args.tokens, args.temperature, args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()

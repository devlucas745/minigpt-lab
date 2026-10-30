"""Simple reproducible perplexity evaluation for a trained character-level model."""
import argparse
import math
import torch

from minigpt.data import load_text, split_ids, get_batch
from minigpt.generate import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/minigpt.pt")
    parser.add_argument("--data", required=True, help="Evaluation text from the same domain.")
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    text = load_text(args.data)
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    if data.numel() <= model.block_size:
        parser.error("Evaluation corpus must be longer than model block_size.")
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = get_batch(data, min(32,  data.numel() - model.block_size), model.block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    print({"mean_cross_entropy": mean_loss, "perplexity": math.exp(mean_loss), "batches": len(losses)})


if __name__ == "__main__":
    main()

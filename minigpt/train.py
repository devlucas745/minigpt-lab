import argparse
import json
import random
import time
from pathlib import Path

import torch

from minigpt.data import get_batch, load_text, split_ids
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer


def estimate_loss(model, train_ids, val_ids, batch_size, block_size, device, eval_iters=20):
    model.eval()
    result = {}
    with torch.no_grad():
        for name, split in (("train", train_ids), ("val", val_ids)):
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(split, batch_size, block_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description="Train MiniGPT on a plain-text corpus.")
    parser.add_argument("--data", default="data/tiny_corpus.txt")
    parser.add_argument("--out", default="artifacts/minigpt.pt")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.steps < 1 or args.eval_interval < 1:
        parser.error("--steps and --eval-interval must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = load_text(args.data)
    tokenizer = CharTokenizer.from_text(text)
    all_ids = tokenizer.encode(text)
    train_ids, val_ids = split_ids(all_ids)
    if train_ids.numel() <= args.block_size or val_ids.numel() <= args.block_size:
        parser.error("Corpus must be longer than block_size in both train and validation splits.")

    model = MiniGPT(
        tokenizer.vocab_size, args.block_size, args.n_embd, args.n_head,
        args.n_layer, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(
                model, train_ids, val_ids, args.batch_size, args.block_size, device
            )
            row = {"step": step, **losses, "elapsed_seconds": time.perf_counter() - start}
            history.append(row)
            print(json.dumps(row))

        x, y = get_batch(train_ids, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": tokenizer.vocab_size, "block_size": args.block_size,
            "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
            "dropout": args.dropout,
        },
        "tokenizer": {"stoi": tokenizer.stoi, "itos": tokenizer.itos},
        "training": {"seed": args.seed, "steps": args.steps, "device": str(device)},
        "history": history,
    }, out)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {out}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()

# MiniGPT Lab

**An educational GPT-style language model built from scratch in PyTorch, with training, evaluation, and an inference API.**

MiniGPT Lab is a reproducible experiment in decoder-only Transformers. It includes causal multi-head self-attention, positional embeddings, residual Transformer blocks, next-token training, text generation, loss/perplexity evaluation, automated tests, and an optional FastAPI service.

> **Scope:** this is a small educational model, not a production-grade assistant. The included tiny corpus exists only to smoke-test the training pipeline. Meaningful language quality requires a larger, licensed, domain-appropriate corpus and more compute. Do not compare its output quality with commercial LLMs without a controlled evaluation.

## Features

- GPT-style decoder-only Transformer implemented in PyTorch
- Causal scaled dot-product attention
- Character-level tokenizer for a transparent baseline
- Next-token prediction and AdamW optimization
- Gradient clipping and train/validation loss logging
- Checkpoint with model configuration and vocabulary
- Sampling with temperature and top-k
- Evaluation using cross-entropy and perplexity
- FastAPI `/health` and `/generate` endpoints
- Unit tests for tokenization, tensor shapes, generation, and data batching

## Architecture

```text
Text corpus
   │
   ▼
Character tokenizer ──► token IDs
                           │
                           ▼
              Token + positional embeddings
                           │
                           ▼
          [LayerNorm → Causal Self-Attention → Residual]
                           │
                           ▼
                 [LayerNorm → MLP → Residual]
                    repeated N layers
                           │
                           ▼
                LayerNorm → vocabulary logits
                           │
                           ▼
             next-token cross-entropy objective
```

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Train a smoke-test model on the included tiny corpus:

```bash
python -m minigpt.train --data data/tiny_corpus.txt --steps 100 --eval-interval 25 --block-size 32
```

Generate text:

```bash
python -m minigpt.generate --checkpoint artifacts/minigpt.pt --prompt "A transformer" --tokens 100
```

The corpus has a small vocabulary. Use a prompt containing only characters present in the training corpus; otherwise the tokenizer raises a clear error.

## Training on a real corpus

1. Obtain a corpus you are legally allowed to use and redistribute, if you plan to publish it.
2. Place the text locally, for example at `data/my_corpus.txt`. This path is git-ignored by default.
3. Choose a block size that fits the dataset and available memory.
4. Train and preserve the logs and checkpoint metadata.

```bash
python -m minigpt.train \
  --data data/my_corpus.txt \
  --out artifacts/experiment_001.pt \
  --steps 5000 \
  --eval-interval 250 \
  --batch-size 32 \
  --block-size 128 \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 4 \
  --seed 1337
```

For meaningful experiments, keep a held-out validation split, avoid test-set tuning, run multiple seeds when feasible, and report the exact corpus, hardware, configuration, wall-clock time, and limitations. Character-level perplexity is not directly comparable to token-level perplexity from another tokenizer.

## Evaluation

```bash
python -m minigpt.evaluate --checkpoint artifacts/experiment_001.pt --data data/heldout.txt --batches 100
```

The evaluation corpus must use characters in the trained vocabulary and should be held out from training. The current evaluator reports approximate cross-entropy and perplexity from random windows. For publication-quality results, also add fixed-window evaluation, confidence intervals across seeds, and benchmark prompts shared by all models.

## API

Train a checkpoint first, then run:

```bash
MINIGPT_CHECKPOINT=artifacts/minigpt.pt uvicorn minigpt.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Generate:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A transformer","max_new_tokens":40,"temperature":0.8,"top_k":10}'
```

The API loads the checkpoint at startup. It returns HTTP 503 if no checkpoint is available and HTTP 422 if the prompt contains unknown characters.

**Security note:** checkpoints loaded with `torch.load` must be trusted. Do not expose this experimental API publicly without authentication, rate limits, request limits, and deployment hardening.

## Comparing against an existing model

Install optional Hugging Face support:

```bash
pip install -e ".[hf]"
```

A fair comparison should define a task-specific dataset, fixed prompts, decoding settings, evaluation metrics, model version, hardware, and latency measurement. Compare the MiniGPT against a small pretrained causal language model available to your environment, or a hosted API if credentials and terms permit. Record results rather than writing claims in advance.

Suggested comparison table:

| Measure | MiniGPT | Pretrained baseline |
|---|---:|---:|
| Parameters | Measure locally | Record from model config |
| Training cost | Measure locally | Usually report as not reproduced |
| Validation loss / perplexity | Measure on same compatible tokenization | Only compare if tokenization/objective are compatible |
| Median latency | Benchmark | Benchmark |
| Output quality | Blind human rubric or task metric | Same rubric |
| License / usage constraints | This code's license | Record model license and terms |

## Reproducibility checklist

- [ ] Document corpus source, license, and preprocessing.
- [ ] Save seed and all hyperparameters.
- [ ] Keep train/validation/test splits separate.
- [ ] Save training logs and evaluation outputs.
- [ ] Report hardware and elapsed time.
- [ ] Run tests on CPU and, if available, GPU.
- [ ] Include failure cases and limitations in the report.
- [ ] Never claim results that have not been measured.

## Roadmap

- [ ] Add BPE tokenizer support
- [ ] Add learning-rate warmup and cosine decay
- [ ] Add mixed precision and gradient accumulation
- [ ] Add checkpoint resume and early stopping
- [ ] Add experiment tracking and loss curves
- [ ] Add a Hugging Face model adapter for controlled comparisons
- [ ] Add Dockerfile and CI workflow
- [ ] Add benchmark suite and model card

## Project structure

```text
minigpt-lab/
├── data/
│   └── tiny_corpus.txt
├── minigpt/
│   ├── api.py
│   ├── data.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_tokenizer.py
├── pyproject.toml
└── README.md
```

## License

Choose a license before making the repository public. The code can be released under MIT if that fits your intent; separately verify the rights and license for every training corpus and pretrained model used in experiments.

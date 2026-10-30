"""Optional inference API for a locally trained MiniGPT checkpoint."""
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from minigpt.generate import load_checkpoint

app = FastAPI(title="MiniGPT Lab API", version="0.1.0")
CHECKPOINT = os.getenv("MINIGPT_CHECKPOINT", "artifacts/minigpt.pt")
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1)


@app.on_event("startup")
def startup():
    global model, tokenizer
    if os.path.exists(CHECKPOINT):
        model, tokenizer = load_checkpoint(CHECKPOINT, device)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/generate")
def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Checkpoint not found. Train MiniGPT first.")
    try:
        ids = torch.tensor([tokenizer.encode(request.prompt)], dtype=torch.long, device=device)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    output = model.generate(ids, request.max_new_tokens, request.temperature, request.top_k)
    return {"text": tokenizer.decode(output[0].tolist()), "model": "MiniGPT"}

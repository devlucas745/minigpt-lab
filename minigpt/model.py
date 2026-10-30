import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention; future tokens are masked."""
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = channels // self.n_head
        q = q.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, head_dim).transpose(1, 2)

        # PyTorch scaled_dot_product_attention applies the causal mask efficiently.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only GPT-style model trained with next-token prediction."""
    def __init__(
        self, vocab_size: int, block_size: int = 128, n_embd: int = 128,
        n_head: int = 4, n_layer: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        if vocab_size < 2 or block_size < 1 or n_layer < 1:
            raise ValueError("vocab_size >= 2, block_size >= 1 and n_layer >= 1 required")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        # Weight tying reduces parameters and is common in language models.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, sequence]")
        if idx.size(1) > self.block_size:
            raise ValueError(f"Sequence length {idx.size(1)} exceeds block_size {self.block_size}")
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        logits = self.lm_head(self.final_ln(self.blocks(x)))
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx")
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = idx[:, -self.block_size:]
                logits, _ = self(context)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits[logits < values[:, [-1]]] = -torch.inf
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)
        return idx

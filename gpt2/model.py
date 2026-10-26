"""GPT-2 style language model, built from the same pieces as attention.ipynb."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304  # GPT-2's 50257 tokens, padded to a multiple of 64 so the GPU matmuls run faster
    context_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    """All heads in one class: one projection for Q, K and V, then split into heads with view/transpose."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)  # W_q, W_k, W_v side by side, like SelfAttentionBest
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)  # mixes the heads and maps back to d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        queries, keys, values = self.qkv(x).split(d_model, dim=-1)  # each (B, T, d_model)
        # (B, T, d_model) -> (B, n_heads, T, head_dim), so every head is attended in one batched call
        queries, keys, values = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (queries, keys, values))
        z = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )                                                          # (B, n_heads, T, head_dim)
        z = z.transpose(1, 2).reshape(B, T, d_model)               # join the heads back together
        return self.out_proj(z)


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.fc_out = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x), approximate="tanh"))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections: this is why attention and feed-forward must keep the d_model size
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying, as in GPT-2: the same matrix embeds and un-embeds tokens

        # GPT-2 initialization. Small normal weights (not torch.rand's [0, 1), which makes deep nets diverge), and the
        # projections that write into the residual stream are scaled down so the sum over 2 * n_layers of them stays small.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "fc_out.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # the tied weight is counted once

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        T = idx.shape[1]
        assert T <= self.cfg.context_length, f"sequence of {T} tokens is longer than context_length"
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))                                            # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = 50,
                 n_vocab: int | None = None) -> torch.Tensor:
        """Append max_new_tokens sampled tokens to idx (B, T). n_vocab drops the padding ids the tokenizer can't decode."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.context_length:])
            logits = logits[:, -1, :n_vocab] / temperature  # only the prediction for the next token
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, -torch.inf)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_layer=4, n_head=4, dropout=0.2, block_size=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Block size depășit"
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

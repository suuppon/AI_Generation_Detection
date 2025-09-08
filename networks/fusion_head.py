# networks/fusion_head.py
# -*- coding: utf-8 -*-
from typing import Optional, Literal
import torch
import torch.nn as nn


class TopKWeightedTowerFusion(nn.Module):
    """
    입력:
      emb_bte         : (B, T, E)  - 타워별 클립 임베딩
      tower_scores_bt : (B, T)     - 타워 신뢰 점수(예: 각 타워 Top-K 평균 확률). 없으면 평균 풀링
    출력:
      logit_b         : (B,)       - 최종 클립 로짓
    옵션:
      pool            : "sum" | "mean" | "attn"
    """
    def __init__(
        self,
        embed_dim: int,
        hidden: int = 512,
        dropout: float = 0.1,
        use_gate: bool = True,
        pool: Literal["sum", "mean", "attn"] = "sum",
        attn_heads: int = 4,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.pool = pool

        self.proj = nn.Linear(embed_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

        if self.pool == "attn":
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden, num_heads=attn_heads, batch_first=True
            )
            self.attn_ln = nn.LayerNorm(hidden)

        self.out = nn.Linear(hidden, 1)

        self.register_buffer("tower_inv", torch.tensor(1.0 / 4.0))

    def forward(self, emb_bte: torch.Tensor, tower_scores_bt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        emb_bte: (B,T,E), tower_scores_bt: (B,T) or None
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, T, E = emb_bte.shape
        emb_bte = emb_bte.to(device=device)
        x = self.proj(emb_bte)    # (B,T,H)
        x = self.norm(x)
        x = self.act(self.drop(x))  # (B,T,H)

        if self.pool == "attn":
            x2, _ = self.attn(x, x, x)   # (B,T,H)
            x = self.attn_ln(x + x2)

        if self.use_gate and (tower_scores_bt is not None):
            gate = torch.softmax(tower_scores_bt, dim=1).unsqueeze(-1) 
            gate = gate.to(device) # (B,T,1)
            x = (x * gate).sum(dim=1)  # (B,H)
        
        else:
            if self.pool in ("attn", "mean"):
                x = x.mean(dim=1)      # (B,H)
            else:  # "sum"
                x = x.sum(dim=1) * self.tower_inv  # (B,H)

        logit = self.out(x).squeeze(1)  # (B,)
        return logit
# networks/multi_tower.py
import copy
from typing import List, Optional, Dict, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_head import TopKWeightedTowerFusion  # ★ Fusion Head 내장 사용

Tensor = torch.Tensor

def _parse_devices(tower_devices: str, fallback_device: torch.device):
    if not tower_devices:
        return None
    ids = [s.strip() for s in tower_devices.split(',') if s.strip() != '']
    if not ids:
        return None
    return [torch.device(f'cuda:{int(x)}') for x in ids]


class MultiTowerFromCloned(nn.Module):
    """
    동일한 base_model을 N개 clone (N>=1). dict/list/tensor 입력 지원.
    내부에 TopK 임베딩 집계 + Fusion Head까지 포함.
    호출: forward(inputs) → (B,) fusion logits
    부가 산출물 필요 시 return_aux=True로 받아갈 수 있음.
    """

    def __init__(self,
                 base_model: nn.Module,
                 n_towers: int = 2,
                 num_classes: Optional[int] = 1,     # binary 가정
                 tower_names: Optional[List[str]] = None,
                 tower_device_map: Optional[Dict[str, torch.device]] = None,
                 tower_devices: Optional[List[torch.device]] = None,
                 main_device: Optional[torch.device] = None,
                 # Top-K & Fusion 설정
                 topk: int = 3,
                 embed_dim: int = 512,
                 fusion_hidden: int = 512,
                 fusion_dropout: float = 0.1,
                 fusion_use_gate: bool = True,
                 fusion_pool: str = "attn"):
        super().__init__()
        assert n_towers >= 1, "n_towers must be >= 1"
        self.topk = int(topk)
        self.main_device = main_device or next(base_model.parameters()).device

        # 타워 이름
        if tower_names is None:
            tower_names = [f"tower{i}" for i in range(n_towers)]
        assert len(tower_names) == n_towers and len(set(tower_names)) == len(tower_names)
        self.tower_names: List[str] = tower_names
        self._name_to_idx = {name: i for i, name in enumerate(self.tower_names)}

        # clone towers
        self.towers = nn.ModuleList([copy.deepcopy(base_model) for _ in range(n_towers)])
        self.num_classes: Optional[int] = num_classes

        # 디바이스 배치
        if tower_device_map is not None:
            self.tower_device_map = {k: v for k, v in tower_device_map.items()}
            for name, mod in zip(self.tower_names, self.towers):
                dev = self.tower_device_map.get(name, self.main_device)
                mod.to(dev)
        elif tower_devices is not None:
            assert len(tower_devices) >= len(self.towers)
            for mod, dev in zip(self.towers, tower_devices):
                mod.to(dev)
            self.tower_device_map = {name: dev for name, dev in zip(self.tower_names, tower_devices)}
        else:
            for mod in self.towers:
                mod.to(self.main_device)
            self.tower_device_map = {name: self.main_device for name in self.tower_names}

        # ★ Fusion Head 내장
        self.fusion_head = TopKWeightedTowerFusion(
            embed_dim=embed_dim,
            n_towers=n_towers,
            hidden=fusion_hidden,
            dropout=fusion_dropout,
            use_gate=fusion_use_gate,
            pool=fusion_pool,
        ).to(self.main_device)

    # ---- 내부 헬퍼 ----
    def _forward_frames_one(self, x_bfchw: Tensor, tower: nn.Module) -> Tensor:
        B, F, C, H, W = x_bfchw.shape
        x_bt = x_bfchw.view(B * F, C, H, W).to(next(tower.parameters()).device, non_blocking=True)
        out_bt = tower(x_bt)                 # (B*F,1) 가정
        return out_bt.view(B, F).to(self.main_device, non_blocking=True)

    def _logits_and_embeddings_one(self, x_bfchw: Tensor, tower: nn.Module) -> Tuple[Tensor, Tensor]:
        B, F, C, H, W = x_bfchw.shape
        dev = next(tower.parameters()).device
        x_bt = x_bfchw.view(B * F, C, H, W).to(dev, non_blocking=True)
        logits_bf = tower(x_bt).view(B, F).to(self.main_device, non_blocking=True)  # (B,F)
        embeds_be = tower.get_embedding(x_bt)                                       # (B*F,E)
        embeds_bfe = embeds_be.view(B, F, -1).to(self.main_device, non_blocking=True)  # (B,F,E)
        return logits_bf, embeds_bfe

    def _build_fusion_inputs(self, inputs: List[Tensor], k: int) -> Tuple[Tensor, Tensor]:
        """
        모든 타워(단일 포함)에 대해 Top-K 임베딩 가중합(clip_embed)과 score 계산.
        returns: emb_bte (B,T,E), scores_bt (B,T)
        """
        clip_embeds, tower_scores = [], []
        for x_bfchw, tower in zip(inputs, self.towers):
            logits_bf, embeds_bfe = self._logits_and_embeddings_one(x_bfchw, tower)
            probs_bf = torch.sigmoid(logits_bf)
            kk = max(1, min(k, probs_bf.size(1)))
            topk_vals, topk_idx = torch.topk(probs_bf, k=kk, dim=1)         # (B,k)
            w = torch.softmax(topk_vals, dim=1).unsqueeze(-1)               # (B,k,1)
            idx = topk_idx.unsqueeze(-1).expand(-1, -1, embeds_bfe.size(-1))  # (B,k,E)
            topk_embeds = torch.gather(embeds_bfe, dim=1, index=idx)        # (B,k,E)
            clip_embed_be = (topk_embeds * w).sum(dim=1)                    # (B,E)
            tower_score_b = topk_vals.mean(dim=1)                           # (B,)
            clip_embeds.append(clip_embed_be)
            tower_scores.append(tower_score_b)
        emb_bte = torch.stack(clip_embeds, dim=1)    # (B,T,E)
        scores_bt = torch.stack(tower_scores, dim=1) # (B,T)
        return emb_bte, scores_bt

    # ---- 공개 forward ----
    def forward(self,
                inputs: Union[List[Tensor], Dict[str, Tensor], Tensor],
                return_aux: bool = False) -> Union[Tensor, Dict[str, Tensor]]:
        """
        inputs: dict(listed by tower_names) | list[Tensor] | Tensor(shared)
                각 텐서는 (B,F,C,H,W) 또는 (B,C,H,W).
        반환:
          - return_aux=False: fusion_logit_b (B,)
          - return_aux=True : {"logit_b":(B,), "emb_bte":(B,T,E), "scores_bt":(B,T)}
        """
        # 입력 정규화 → list[Tensor] (그리고 (B,C,H,W) → (B,1,C,H,W))
        if isinstance(inputs, dict):
            xs = [inputs[name] for name in self.tower_names]
        elif torch.is_tensor(inputs):
            xs = [inputs for _ in self.towers]
        else:
            xs = list(inputs)
            assert len(xs) == len(self.towers), "len(inputs) must equal number of towers"
        for i in range(len(xs)):
            if xs[i].dim() == 4:
                xs[i] = xs[i].unsqueeze(1)

        B, F = xs[0].shape[:2]
        k = max(1, min(self.topk, F))

        emb_bte, scores_bt = self._build_fusion_inputs(xs, k)  # (B,T,E), (B,T)
        logit_b = self.fusion_head(emb_bte, scores_bt)         # (B,)

        if return_aux:
            return {"logit_b": logit_b, "emb_bte": emb_bte, "scores_bt": scores_bt}
        return logit_b
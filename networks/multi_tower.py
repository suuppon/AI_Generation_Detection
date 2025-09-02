# networks/multi_tower.py
import copy
from typing import List, Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

def _parse_devices(tower_devices: str, fallback_device: torch.device):
    """
    '0,1,2' 형태의 문자열을 받아 장치 리스트를 반환.
    비어있으면 None(=모든 타워를 fallback_device에 배치).
    """
    if not tower_devices:
        return None
    ids = [s.strip() for s in tower_devices.split(',') if s.strip() != '']
    if not ids:
        return None
    return [torch.device(f'cuda:{int(x)}') for x in ids]

class MultiTowerFromCloned(nn.Module):
    """
    동일한 base_model을 N개 clone하여 feature별로 독립 학습.
    입력:
      - dict 모드: {"edge":[B,T,C,H,W], "texture":[B,T,C,H,W], ...} (features_order와 1:1 매핑)
      - concat/공유 모드: [B,T,C,H,W] → 모든 타워에 동일 입력

    내부 처리 trick:
      - 시계열을 유지한 채, forward에서만 [B*T, C, H, W]로 펴서 모델을 한 방에 태움 → [B,T,num_classes]로 복원.

    결합:
      - 'avg_logits': 타워별 logits 시퀀스 평균 → probs_seq = softmax(logits_seq)
      - 'softvote' : 타워별 probs 시퀀스 평균 → logits_seq = log(probs_seq)
    반환 키:
      {
        "logits_seq": [B,T,C],     # 최종(타워 결합 후) 시퀀스 로짓
        "probs_seq":  [B,T,C],     # 최종(타워 결합 후) 시퀀스 확률
        "logits":      [B,C],      # 시간 평균 집계(훈련용 편의)
        "probs":       [B,C],      # 시간 평균 집계(훈련용 편의)
        "towers": {
          <feature or idx>: {
            "logits_seq": [B,T,C],
            "probs_seq":  [B,T,C],
            "topk_idx":   [B,T,K],
            "topk_val":   [B,T,K],
          }, ...
        }
      }
    """
class MultiTowerFromCloned(nn.Module):
    def __init__(self,
                 base_model: nn.Module,
                 n_towers: int = 2,
                 num_classes: Optional[int] = None,
                 combine: str = 'avg_logits',
                 features_order: Optional[List[str]] = None,
                 tower_devices: Optional[List[torch.device]] = None,
                 main_device: Optional[torch.device] = None):
        super().__init__()
        assert n_towers >= 2, "n_towers must be >= 2"
        self.combine = combine
        self.main_device = main_device or next(base_model.parameters()).device
        self.features_order = features_order

        # clone towers
        self.towers = nn.ModuleList([copy.deepcopy(base_model) for _ in range(n_towers)])

        # place towers on devices (optional true model-parallel)
        self.tower_devices = tower_devices   # << 여기서 Trainer에서 넘겨준 리스트 그대로 사용
        if self.tower_devices is not None:
            assert len(self.tower_devices) >= len(self.towers), \
                "Provide at least as many devices as towers"
            for mod, dev in zip(self.towers, self.tower_devices):
                mod.to(dev)
        else:
            for mod in self.towers:
                mod.to(self.main_device)

        # concat 모드일 때 Linear head 추가 (여긴 그대로 두셔도 됩니다)
        if self.combine == 'concat':
            if num_classes is None:
                raise ValueError("num_classes must be provided when using 'concat'")
            raise NotImplementedError("concat head not implemented")

    def _forward_seq_tower(self, tower: nn.Module, x: Tensor, dev: Optional[torch.device]) -> Tensor:
        """
        x: [B,T,C,H,W] 또는 [B,C,H,W]
        return: logits_seq [B,T,num_classes]  (T가 없으면 T=1로 간주)
        """
        if x.dim() == 4:
            # [B,C,H,W] -> [B,1,C,H,W]
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape

        x_bt = x.reshape(B * T, C, H, W)
        if dev is not None:
            x_bt = x_bt.to(dev, non_blocking=True)
            out_bt = tower(x_bt)               # [B*T, num_classes]
            out_bt = out_bt.to(self.main_device, non_blocking=True)
        else:
            out_bt = tower(x_bt)

        out_seq = out_bt.view(B, T, self.num_classes)  # [B,T,C]
        return out_seq

    def forward(self, x: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        per_tower: Dict[str, Dict[str, Tensor]] = {}
        logits_seq_list = []
        probs_seq_list = []

        # 입력 매핑
        if isinstance(x, dict):
            assert self.features_order is not None, "features_order must be provided for dict input"
            assert len(self.features_order) == len(self.towers), "features_order size must equal n_towers"
            it = enumerate(self.features_order)
            get_feature = lambda key: x[key]  # noqa: E731
        else:
            it = enumerate(range(len(self.towers)))
            get_feature = lambda _key: x      # noqa: E731

        for i, key in it:
            dev = None if self.tower_devices is None else self.tower_devices[i]
            feats = get_feature(key)                          # [B,T,C,H,W] or [B,C,H,W]
            logits_seq_i = self._forward_seq_tower(self.towers[i], feats, dev)  # [B,T,C]
            probs_seq_i = F.softmax(logits_seq_i, dim=-1)     # [B,T,C]

            # per-tower top-k (타임스텝별)
            topk_val, topk_idx = torch.topk(probs_seq_i, k=self.topk, dim=-1)  # [B,T,K]

            per_tower[str(key)] = {
                "logits_seq": logits_seq_i,
                "probs_seq": probs_seq_i,
                "topk_idx": topk_idx,
                "topk_val": topk_val,
            }
            logits_seq_list.append(logits_seq_i)
            probs_seq_list.append(probs_seq_i)

        # 타워 결합 (타임스텝별)
        if self.combine == 'avg_logits':
            logits_seq = torch.stack(logits_seq_list, dim=0).mean(dim=0)  # [B,T,C]
            probs_seq = F.softmax(logits_seq, dim=-1)
        else:  # 'softvote'
            probs_seq = torch.stack(probs_seq_list, dim=0).mean(dim=0)    # [B,T,C]
            logits_seq = torch.log(probs_seq.clamp_min(1e-8))             # 안정적 logit 대체

        # 시간 축 집계(편의): 평균
        logits = logits_seq.mean(dim=1)    # [B,C]
        probs = probs_seq.mean(dim=1)      # [B,C]

        return {
            "logits_seq": logits_seq,      # [B,T,C]
            "probs_seq": probs_seq,        # [B,T,C]
            "logits": logits,              # [B,C]  (time-avg)
            "probs": probs,                # [B,C]  (time-avg)
            "towers": per_tower
        }
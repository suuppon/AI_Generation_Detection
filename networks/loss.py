# loss.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
ReduceMode = Literal["lsep", "mean"]
TowerCombine = Literal["mean", "sum"]
FocalType = Literal["none", "sigmoid_focal"]

def _ensure_1d(y: Tensor) -> Tensor:
    # y: (B,) or (B,1) -> (B,)
    return y.view(-1)

def _safe_log(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.log(torch.clamp(x, min=eps))

def select_topk(
    x: Tensor, k: int, dim: int = -1, largest: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    x: (..., F, ...)
    반환:
      topk_vals: (..., k, ...)
      topk_idx : (..., k, ...)
    """
    k = max(1, min(k, x.size(dim)))
    topk_vals, topk_idx = torch.topk(x, k=k, dim=dim, largest=largest, sorted=False)
    return topk_vals, topk_idx


def tower_clip_logit_from_frames(
    frame_logits_bf: Tensor,  # (B, F) — 프레임단 로짓
    topk: int,
    reduce: ReduceMode = "lsep",
) -> Tensor:
    """
    타워 1개에 대해 (B,F) -> (B,) 클립 로짓으로 축약.
    - 'mean' : Top-K 로짓 평균
    - 'lsep' : log( sum(exp(Top-K))/k ) (LogSumExp-평균; max의 부드러운 근사)
    """
    k = max(1, min(topk, frame_logits_bf.size(1)))
    tk, _ = select_topk(frame_logits_bf, k=k, dim=1, largest=True)  # (B,k)

    if reduce == "mean":
        return tk.mean(dim=1)  # (B,)
    elif reduce == "lsep":
        return torch.logsumexp(tk, dim=1) - torch.log(
            torch.tensor(k, device=tk.device, dtype=tk.dtype)
        )
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")


def towers_soft_combine(
    tower_clip_logits_bt: Tensor,  # (B, T)
    how: TowerCombine = "mean",
) -> Tensor:
    """
    학습 시 타워 간 soft 결합 (미분 가능).
    - 'mean': (B,)
    - 'sum' : (B,)
    """
    if how == "mean":
        return tower_clip_logits_bt.mean(dim=1)
    elif how == "sum":
        return tower_clip_logits_bt.sum(dim=1)
    else:
        raise ValueError(f"Unknown tower combine: {how}")


def sigmoid_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    alpha: Optional[float] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """
    logits: (B,)
    targets: (B,) in {0,1}
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)  # p_t = p for y=1 else (1-p)
    focal = (1 - p_t) ** gamma * ce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal = alpha_t * focal
    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    return focal


class ClipLevelLoss(nn.Module):
    """
    비디오 클립 단위 이진 분류용 Loss 모듈.
    - 타워별 프레임 로짓 -> Top-K -> 타워별 클립 로짓
    - 학습 시 타워 간 soft 결합(평균/합) -> 최종 클립 로짓
    - 최종 로짓에 BCEWithLogits 또는 Focal Loss 적용
    - (선택) 정규화 항:
        * tower一致성(consistency) 정규화
        * 예측 엔트로피/스파스니 정규화
    """

    def __init__(
        self,
        *,
        topk: int = 5,
        frame_reduce: ReduceMode = "lsep",
        tower_combine: TowerCombine = "mean",
        focal: FocalType = "none",
        focal_alpha: Optional[float] = None,
        focal_gamma: float = 2.0,
        bce_pos_weight: Optional[float] = None,  # class imbalance용
        label_smoothing: float = 0.0,            # [0..1)
        # 정규화 계수들 (0이면 꺼짐)
        lambda_tower_consistency: float = 0.0,
        lambda_entropy: float = 0.0,
    ):
        super().__init__()
        self.topk = topk
        self.frame_reduce = frame_reduce
        self.tower_combine = tower_combine
        self.focal = focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = float(label_smoothing)

        if bce_pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(bce_pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

        self.lambda_tower_consistency = float(lambda_tower_consistency)
        self.lambda_entropy = float(lambda_entropy)

    def _bce_with_logits(
        self, logits_b: Tensor, targets_b: Tensor
    ) -> Tensor:
        # label smoothing (간단 버전): y' = y*(1-ε) + 0.5*ε
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets_b = targets_b * (1.0 - eps) + 0.5 * eps

        return F.binary_cross_entropy_with_logits(
            logits_b, targets_b,
            reduction="mean",
            pos_weight=self.pos_weight,
        )

    def _main_loss(
        self, final_clip_logits_b: Tensor, y_b: Tensor
    ) -> Tensor:
        if self.focal == "sigmoid_focal":
            return sigmoid_focal_loss_with_logits(
                final_clip_logits_b, y_b,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="mean",
            )
        else:
            return self._bce_with_logits(final_clip_logits_b, y_b)

    @staticmethod
    def _tower_consistency_penalty(
        tower_clip_probs_bt: Tensor,
    ) -> Tensor:
        """
        타워 간 예측 확률의 분산/산포를 줄이는 정규화 (소프트).
        간단히 각 샘플에 대해 타워 표준편차를 벌점으로 사용.
        """
        # (B,T)
        std_t = tower_clip_probs_bt.std(dim=1)  # (B,)
        return std_t.mean()

    @staticmethod
    def _entropy_penalty(
        final_clip_probs_b: Tensor,
    ) -> Tensor:
        """
        예측의 엔트로피를 낮추는 정규화(선택).
        """
        p = torch.clamp(final_clip_probs_b, 1e-6, 1 - 1e-6)
        ent = -(p * _safe_log(p) + (1 - p) * _safe_log(1 - p))  # (B,)
        return ent.mean()

    def forward(
        self,
        tower_frame_logits: Union[List[Tensor], Tensor],
        labels: Tensor,
        # 선택: 가변 길이 클립을 위한 마스크 (True=유효). shape은 tower_frame_logits와 호환되게.
        frame_mask: Optional[Union[List[Tensor], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
          tower_frame_logits:
            - List[Tensor(B,F)]  (타워 리스트)
            - 또는 Tensor(B,T,F)  (타워 축 포함)
          labels: Tensor(B,) in {0,1}
          frame_mask (선택):
            - List[Tensor(B,F)] 또는 Tensor(B,T,F) (True/1: 유효 프레임)
            - 마스크가 있으면, 유효하지 않은 프레임 로짓을 매우 작은 값으로 덮어 Top-K에서 배제
        Returns:
          dict(loss=..., main=..., tower_consistency=..., entropy=..., aux={...})
        """
        y = _ensure_1d(labels).float()  # (B,)

        # 1) 입력 정규화: List[ (B,F) ] 형태로 맞추기
        if isinstance(tower_frame_logits, list):
            logits_list: List[Tensor] = tower_frame_logits
        else:
            assert tower_frame_logits.dim() == 3, "Expect (B,T,F)"
            logits_list = [tower_frame_logits[:, t, :] for t in range(tower_frame_logits.size(1))]

        T = len(logits_list)
        B = y.size(0)

        # 2) 마스크 처리(있다면): 유효하지 않은 프레임은 -inf 로짓으로 만들어 Top-K에서 배제
        if frame_mask is not None:
            if isinstance(frame_mask, list):
                mask_list = frame_mask
            else:
                assert frame_mask.dim() == 3, "mask must be (B,T,F)"
                mask_list = [frame_mask[:, t, :] for t in range(frame_mask.size(1))]
            assert len(mask_list) == T
        else:
            mask_list = [None] * T

        # 3) 타워별 클립 로짓 계산: (B,F) -> (B,)
        tower_clip_logits_bt: List[Tensor] = []
        tower_clip_probs_bt: List[Tensor] = []  # 정규화 항/리포팅용

        for t in range(T):
            logits_bf = logits_list[t]  # (B,F)

            m = mask_list[t]
            if m is not None:
                # True(1)=유효, False(0)=무시 → 무시 프레임은 매우 작은 로짓으로 다운
                # down_value는 음의 큰 값으로 해 Top-K에서 선택되지 않게 함
                down_value = torch.finfo(logits_bf.dtype).min / 8.0
                logits_bf = torch.where(m > 0, logits_bf, logits_bf.new_full((), down_value))

            clip_logit_t = tower_clip_logit_from_frames(
                logits_bf, topk=self.topk, reduce=self.frame_reduce
            )  # (B,)
            tower_clip_logits_bt.append(clip_logit_t)

            # 확률도 저장(정규화 항/리포트)
            # 프레임 확률 → Top-K 평균 확률(참고용)
            with torch.no_grad():
                probs_bf = torch.sigmoid(logits_bf)
                k = max(1, min(self.topk, probs_bf.size(1)))
                topk_p, _ = select_topk(probs_bf, k=k, dim=1, largest=True)
                p_clip_t = topk_p.mean(dim=1)  # (B,)
                tower_clip_probs_bt.append(p_clip_t)

        tower_clip_logits_bt = torch.stack(tower_clip_logits_bt, dim=1)  # (B,T)
        tower_clip_probs_bt = torch.stack(tower_clip_probs_bt, dim=1)    # (B,T), no grad

        # 4) 학습용 최종 로짓(타워 soft 결합)
        final_clip_logits_b = towers_soft_combine(tower_clip_logits_bt, how=self.tower_combine)  # (B,)

        # 5) 메인 손실
        main = self._main_loss(final_clip_logits_b, y)

        # 6) 정규화 항
        reg_cons = torch.tensor(0.0, device=main.device)
        reg_ent = torch.tensor(0.0, device=main.device)

        if self.lambda_tower_consistency > 0.0 and T > 1:
            reg_cons = self._tower_consistency_penalty(tower_clip_probs_bt) * self.lambda_tower_consistency

        if self.lambda_entropy > 0.0:
            with torch.no_grad():
                final_probs_b = torch.sigmoid(final_clip_logits_b)
            reg_ent = self._entropy_penalty(final_probs_b) * self.lambda_entropy

        loss = main + reg_cons + reg_ent

        # 리포팅용 부가값
        with torch.no_grad():
            final_probs_b = torch.sigmoid(final_clip_logits_b)
            # 추론 시 hard vote와 동일한 로직을 모니터링용으로 계산
            thr = 0.5
            hard_votes = (tower_clip_probs_bt > thr).int()  # (B,T)
            hard_pred = (hard_votes.sum(dim=1) >= (T + 1) // 2).float()  # (B,)

        return {
            "loss": loss,
            "main": main,
            "tower_consistency": reg_cons,
            "entropy": reg_ent,
            "aux": {
                "final_clip_logit": final_clip_logits_b.detach(),   # (B,)
                "final_clip_prob": final_probs_b.detach(),          # (B,)
                "tower_clip_logits": tower_clip_logits_bt.detach(), # (B,T)
                "tower_clip_probs": tower_clip_probs_bt.detach(),   # (B,T)
                "hard_pred": hard_pred,                             # (B,)
            },
        }
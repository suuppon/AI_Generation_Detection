# networks/trainer.py
# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from networks.resnet import resnet50
from networks.base_model import BaseModel
from networks.multi_tower import MultiTowerFromCloned, _parse_devices
from networks.loss import ClipLevelLoss


class Trainer(BaseModel):

    def name(self):
        return "Trainer"

    # ---------------------
    # 초기화 / 모델 구성
    # ---------------------
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        # 0) 하이퍼 파라미터 (옵션에서 가져옴; 기본값 제공)
        self.num_towers = int(getattr(opt, "num_towers", 0)) or max(1, len(getattr(opt, "features", [])))
        self.features_order = getattr(opt, "features", None) if self.num_towers > 1 else None
        self.frames_per_clip: int = int(getattr(opt, "frames_per_clip", 1))
        self.topk: int = int(getattr(opt, "topk", max(1, self.frames_per_clip // 3)))
        self.frame_reduce: str = getattr(opt, "tower_reduce", "lsep")  # "lsep" or "mean"
        self.tower_combine: str = getattr(opt, "tower_combine", "mean")  # "mean" or "sum"

        # focal / class imbalance
        self.use_focal: bool = (getattr(opt, "use_focal", "none") == "sigmoid_focal")
        self.focal_alpha: Optional[float] = getattr(opt, "focal_alpha", None)
        self.focal_gamma: float = float(getattr(opt, "focal_gamma", 2.0))
        self.bce_pos_weight: Optional[float] = getattr(opt, "bce_pos_weight", None)
        self.label_smoothing: float = float(getattr(opt, "label_smoothing", 0.0))

        # regularization
        self.lambda_tower_consistency: float = float(getattr(opt, "lambda_tower_consistency", 0.0))
        self.lambda_entropy: float = float(getattr(opt, "lambda_entropy", 0.0))

        # 1) 기본 단일 타워 백본
        if self.isTrain and not getattr(opt, "continue_train", False):
            base_model = resnet50(pretrained=False, num_classes=1)
        else:
            base_model = resnet50(num_classes=1)

        # 2) 멀티타워 래핑
        main_device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
        devices = _parse_devices(getattr(opt, "tower_devices", ""), main_device)

        if self.num_towers > 1:
            self.model = MultiTowerFromCloned(
                base_model=base_model,
                n_towers=self.num_towers,
                num_classes=1,                       # ★ 추가
                combine="avg_logits",                # 내부 기본 결합(학습 손실은 따로)
                features_order=self.features_order,  # ★ 추가: dict 배치 매핑용
                tower_devices=devices,
                main_device=main_device,
            )
        else:
            self.model = base_model

        self.model.to(self.device)

        # 4) 손실함수 (클립 레벨)
        self.loss_fn = ClipLevelLoss(
            topk=self.topk,
            frame_reduce=self.frame_reduce,            # "lsep" / "mean"
            tower_combine=self.tower_combine,          # "mean" / "sum"
            focal=("sigmoid_focal" if self.use_focal else "none"),
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            bce_pos_weight=self.bce_pos_weight,
            label_smoothing=self.label_smoothing,
            lambda_tower_consistency=self.lambda_tower_consistency,
            lambda_entropy=self.lambda_entropy,
        )

        # 5) Optimizer
        if self.isTrain:
            if getattr(opt, "optim", "adam") == "adam":
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                )
            elif opt.optim == "sgd":
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr,
                    momentum=0.0,
                    weight_decay=0.0,
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # 6) 체크포인트 로드
        if (not self.isTrain) or getattr(opt, "continue_train", False):
            self.load_networks(opt.epoch)

        # 러닝레이트 추적용
        self.lr = getattr(opt, "lr", 1e-3)

    # ---------------------
    # 러닝레이트 스케줄
    # ---------------------
    def adjust_learning_rate(self, min_lr: float = 1e-6):
        for pg in self.optimizer.param_groups:
            pg["lr"] *= 0.9
        new_lr = self.optimizer.param_groups[0]["lr"]
        if new_lr < min_lr:
            return False
        self.lr = new_lr
        print("*" * 25)
        print(f"Changing lr to {self.lr:.6g}")
        print("*" * 25)
        return True

    # ---------------------
    # 입력 주입
    # ---------------------
    def set_input(self, input):
        """
        input: (views, labels)
          - views: dict => {'edge':[B,F,C,H,W], 'texture':...}  (권장)
                  list/tuple => [ (B,F,C,H,W) ] * num_towers
          - labels: Tensor (B,)
        """
        xs, y = input

        if isinstance(xs, dict):
            order = list(xs.keys())
            views = [xs[k] for k in order]
        else:
            views = list(xs)
            assert len(views) == self.num_towers or self.num_towers == 1, "views 길이가 num_towers와 다릅니다."

        self.inputs = [v.to(self.device, non_blocking=True) for v in views]
        self.label = y.to(self.device, non_blocking=True).float()

    # ---------------------
    # 내부 유틸: 타워 하나 forward (B,F,C,H,W) -> (B,F) frame logits
    # ---------------------
    def _tower_forward_frames(self, x_bfchw: torch.Tensor, tower_module: nn.Module) -> torch.Tensor:
        B, F, C, H, W = x_bfchw.shape
        x_bt = x_bfchw.view(B * F, C, H, W)

        tower_dev = next(tower_module.parameters()).device
        x_bt = x_bt.to(tower_dev, non_blocking=True)

        out_bt = tower_module(x_bt)               # (B*F, 1)
        out_bt = out_bt.to(self.device, non_blocking=True)

        return out_bt.view(B, F)                  # (B,F)

    # ---------------------
    # 학습 Forward: frame logits -> loss 구성은 loss_fn에 위임
    # ---------------------
    def forward(self):
        """
        학습 시:
          - 각 타워 t: (B,F,C,H,W) -> frame logits (B,F)
          - loss_fn이 Top-K 집계 + 타워 soft 결합 + BCE/Focal을 수행
        """
        tower_modules = getattr(self.model, "towers", None)
        if tower_modules is None:
            # 단일 타워 (self.model 자체가 백본)
            assert len(self.inputs) == 1, "단일 타워에서는 views 길이가 1이어야 합니다."
            frame_logits_list = [self._tower_forward_frames(self.inputs[0], self.model)]  # [ (B,F) ]
        else:
            assert len(self.inputs) == len(tower_modules), "입력 views와 타워 수가 다릅니다."
            frame_logits_list = [
                self._tower_forward_frames(x_bfchw, tower)
                for x_bfchw, tower in zip(self.inputs, tower_modules)
            ]  # 길이 T, 각 (B,F)

        # loss 계산 (dict 반환)
        ret = self.loss_fn(
            tower_frame_logits=frame_logits_list,    # List[Tensor(B,F)]
            labels=self.label,                        # (B,)
            frame_mask=None,                          # 필요하면 마스크 전달
        )
        # 주 로짓(클립 단)의 모양을 기존 get_loss() 호환 위해 (B,1)로 보관
        self.output = ret["aux"]["final_clip_logit"].unsqueeze(1)  # (B,1)
        self._loss_dict = ret

    def get_loss(self):
        # forward()에서 _loss_dict 채워둠
        return self._loss_dict["loss"]

    # ---------------------
    # 최적화 루프
    # ---------------------
    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad(set_to_none=True)
        self.loss.backward()
        self.optimizer.step()

    # ---------------------
    # 평가/추론용 API (hard vote 포함)
    # ---------------------
    @torch.no_grad()
    def predict_clip(
        self,
        inputs: List[torch.Tensor],      # list length = num_towers, each (B,F,C,H,W)
        prob_threshold: float = 0.5,
        k_top: Optional[int] = None,
        tower_reduce: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        추론:
          - 각 타워: 프레임 확률 -> Top-K 평균 확률 (soft vote)
          - 타워 간 hard vote로 최종 이진 판정
        반환:
          probs_mean: (B,)   # 타워 확률 평균 (soft)
          hard:      (B,)   # hard vote (0/1)
          probs_bt:  (B,T)  # 타워별 확률
        """
        self.model.eval()

        T = self.num_towers
        tower_modules = getattr(self.model, "towers", None)
        if tower_modules is None:
            tower_modules = [self.model]
            assert len(inputs) == 1

        k = int(k_top) if k_top is not None else self.topk
        reduce_mode = tower_reduce if tower_reduce is not None else self.frame_reduce

        tower_probs: List[torch.Tensor] = []
        for x_bfchw, tower in zip(inputs, tower_modules):
            # frame logits (B,F)
            logits_bf = self._tower_forward_frames(x_bfchw.to(self.device), tower)
            # frame -> clip 확률 (Top-K 평균 확률)
            F = logits_bf.size(1)
            kk = max(1, min(k, F))
            probs_bf = torch.sigmoid(logits_bf)                  # (B,F)
            topk_vals, _ = torch.topk(probs_bf, k=kk, dim=1)     # (B,k)
            p_clip_t = topk_vals.mean(dim=1)                     # (B,)
            tower_probs.append(p_clip_t)

        probs_bt = torch.stack(tower_probs, dim=1)                # (B,T)
        probs_mean = probs_bt.mean(dim=1)                         # (B,)
        votes = (probs_bt > prob_threshold).int()                 # (B,T)
        hard = (votes.sum(dim=1) >= (probs_bt.size(1) + 1) // 2).int().float()  # (B,)

        return probs_mean, hard, probs_bt

    # ---------------------
    # 호환: 단일 이미지 입력을 위한 간이 forward (validate 호환성)
    #  - validate()가 self.model(x: (B,C,H,W))을 직접 호출하는 경우 대비
    #  - MultiTowerFromCloned.forward가 4D 입력을 받으면 1번 타워로 처리하도록 구현되어 있다면 불필요
    # ---------------------
    def image_forward_for_validate(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """
        (B,C,H,W) -> (B,1)  (주로 기존 validate()가 이미지 단건 기준일 때 사용)
        """
        self.model.eval()
        tower_modules = getattr(self.model, "towers", None)
        if tower_modules is None:
            return self.model(x_bchw.to(self.device))  # (B,1)
        else:
            # 첫 번째 타워만 사용 (평가 호환 목적)
            return tower_modules[0](x_bchw.to(self.device))  # (B,1)





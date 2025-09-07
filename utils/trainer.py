# utils/trainer.py
# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

from networks.resnet import resnet50
from networks.base_model import BaseModel
from networks.multi_tower import MultiTowerFromCloned, _parse_devices


class Trainer(BaseModel):
    def name(self): return "Trainer"

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

        # ---- 옵션 ----
        self.num_towers = int(getattr(opt, "num_towers", 0)) or max(1, len(getattr(opt, "features", [])))
        self.features_order = getattr(opt, "features", None) if self.num_towers >= 1 else None
        self.frames_per_clip: int = int(getattr(opt, "frames_per_clip", 1))
        self.topk: int = int(getattr(opt, "topk", max(1, self.frames_per_clip // 3)))

        # Fusion 설정 (MultiTower에 전달)
        self.embedding_dim: int = int(getattr(opt, "embedding_dim", 512))
        self.fusion_hidden: int = int(getattr(opt, "fusion_hidden", 512))
        self.fusion_dropout: float = float(getattr(opt, "fusion_dropout", 0.1))
        self.fusion_use_gate: bool = bool(getattr(opt, "fusion_use_gate", True))
        self.fusion_pool: str = str(getattr(opt, "fusion_pool", "attn"))

        # 클래스 불균형 옵션
        self.bce_pos_weight: Optional[float] = getattr(opt, "bce_pos_weight", None)

        # ---- 백본 & 멀티타워(=퓨전 포함) ----
        if self.isTrain and not getattr(opt, "continue_train", False):
            base_model = resnet50(pretrained=False, num_classes=1)
        else:
            base_model = resnet50(num_classes=1)

        main_device = torch.device(f"cuda:{opt.gpu_ids[0]}") if getattr(opt, "gpu_ids", []) else torch.device("cpu")
        devices = _parse_devices(getattr(opt, "tower_devices", ""), main_device)

        self.model = MultiTowerFromCloned(
            base_model=base_model,
            n_towers=self.num_towers,
            num_classes=1,
            tower_names=(self.features_order if self.features_order else None),
            tower_devices=devices,
            main_device=main_device,
            topk=self.topk,
            embed_dim=self.embedding_dim,
            fusion_hidden=self.fusion_hidden,
            fusion_dropout=self.fusion_dropout,
            fusion_use_gate=self.fusion_use_gate,
            fusion_pool=self.fusion_pool,
        ).to(self.device)

        # ---- Loss / Optimizer ----
        pos_w = None if (self.bce_pos_weight is None) else torch.tensor(self.bce_pos_weight, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        if self.isTrain:
            params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            if getattr(opt, "optim", "adam") == "adam":
                self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            elif getattr(opt, "optim", "adam") == "sgd":
                self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=0.0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if (not self.isTrain) or getattr(opt, "continue_train", False):
            self.load_networks(opt.epoch)

        self.lr = getattr(opt, "lr", 1e-3)

    def adjust_learning_rate(self, min_lr: float = 1e-6):
        for pg in self.optimizer.param_groups:
            pg["lr"] *= 0.9
        new_lr = self.optimizer.param_groups[0]["lr"]
        if new_lr < min_lr: return False
        self.lr = new_lr
        print("*"*25); print(f"Changing lr to {self.lr:.6g}"); print("*"*25)
        return True

    def set_input(self, input):
        xs, y = input
        if isinstance(xs, dict):
            order = getattr(self.model, "tower_names", None) or self.features_order or list(xs.keys())
            views = [xs[k] for k in order]
        else:
            views = list(xs)
            assert len(views) == self.num_towers, "views 길이가 num_towers와 다릅니다."
        self.inputs = [v.to(self.device, non_blocking=True) for v in views]
        self.label = y.to(self.device, non_blocking=True).float()

    # -------- Fusion only --------
    def forward(self):
        # MultiTower가 바로 (B,) logits 반환
        logit_b = self.model(self.inputs, return_aux=False)   # (B,)
        loss = self.criterion(logit_b, self.label)
        self.output = logit_b.unsqueeze(1)  # (B,1)
        self._loss = loss

    def get_loss(self): return self._loss

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad(set_to_none=True)
        self.loss.backward()
        self.optimizer.step()

    # ---- 추론 ----
    @torch.no_grad()
    def predict_clip(self, inputs: List[torch.Tensor], prob_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        ret = self.model(inputs, return_aux=False)  # (B,)
        prob_b = torch.sigmoid(ret)
        hard_b = (prob_b > prob_threshold).float()
        return prob_b, hard_b

    @torch.no_grad()
    def predict_clip_fusion(self, inputs: List[torch.Tensor]):
        self.model.eval()
        aux = self.model(inputs, return_aux=True)
        prob_b = torch.sigmoid(aux["logit_b"])
        return prob_b, aux["scores_bt"]
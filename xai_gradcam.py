# xai_gradcam.py
import os
from glob import glob
from typing import List
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2


def _auto_target_layer(model: nn.Module) -> nn.Module:
    """ResNet/MultiTower에서 Grad-CAM 타깃 레이어 자동 탐색."""
    # 1) 일반 ResNet
    try:
        return model.layer4[-1].conv3
    except Exception:
        pass

    # 2) MultiTower 형태
    try:
        return model.towers[0].layer4[-1].conv3
    except Exception:
        pass

    # 3) Fallback: 모델 내 마지막 Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Grad-CAM target layer를 찾지 못했습니다.")
    return last_conv


def _build_transform(no_resize: bool, no_crop: bool):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    ops: List[nn.Module] = []
    if not no_resize:
        ops.append(T.Resize(256))
    if not no_crop:
        ops.append(T.CenterCrop(224))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def _list_images_recursive(folder: str):
    """폴더 아래를 재귀적으로 뒤져 이미지 목록을 수집."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    paths = []
    for ext in exts:
        paths += glob(os.path.join(folder, "**", f"*{ext}"), recursive=True)
    return sorted(paths)


def _pick_logits_from_output(out_obj: object) -> torch.Tensor:
    """
    모델 출력이 dict/tuple/list/tensor 무엇이든 [B] 로짓으로 정규화.
    우선 키: logits, logit, y, scores, (logits_seq/scores_seq/probs_seq는 T-mean)
    """
    if isinstance(out_obj, dict):
        for k in ["logits", "logit", "y", "scores"]:
            if k in out_obj and torch.is_tensor(out_obj[k]):
                t = out_obj[k]
                break
        else:
            for k in ["logits_seq", "scores_seq", "probs_seq"]:
                if k in out_obj and torch.is_tensor(out_obj[k]):
                    t = out_obj[k]
                    if t.dim() == 3:  # [B,T,C]
                        B, T, C = t.shape
                        t = t.mean(dim=1)            # [B,C]
                        t = t.squeeze(1) if C == 1 else t[:, 0]
                    elif t.dim() == 2:               # [B,T]
                        t = t.mean(dim=1)            # [B]
                    return t
            tv = [v for v in out_obj.values() if torch.is_tensor(v)]
            if len(tv) == 1:
                t = tv[0]
            else:
                raise RuntimeError("Cannot find logits in dict output.")
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1); t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    if isinstance(out_obj, (tuple, list)):
        t = None
        for item in out_obj:
            if torch.is_tensor(item):
                t = item
                break
            if isinstance(item, dict):
                return _pick_logits_from_output(item)
        if t is None:
            raise RuntimeError("Cannot find tensor in tuple/list output.")
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1); t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    if torch.is_tensor(out_obj):
        t = out_obj
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1); t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    raise RuntimeError(f"Unsupported output type: {type(out_obj)}")


@torch.no_grad()
def _save_image(path: str, arr_bgr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, arr_bgr)


def save_cams_for_folder(
    model: nn.Module,
    folder: str,
    out_dir: str,
    no_resize: bool = False,
    no_crop: bool = True,
    max_images: int = 16,
):
    """
    폴더(하위 포함) 내 이미지들에 대해 Grad-CAM overlay를 저장합니다.
    - class_idx=0 기준으로 CAM을 계산(이진 분류에서 'fake'를 0으로 가정).
    - validate()가 no_grad로 돌더라도, 여기서는 grad가 활성화되도록 별도 실행.
    """
    device = next(model.parameters()).device
    model.eval()

    target_layer = _auto_target_layer(model)
    tfm = _build_transform(no_resize, no_crop)

    img_paths = _list_images_recursive(folder)[:max_images]
    if not img_paths:
        print(f"[xai] no images found under {folder}")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Hooks
    activations = []
    gradients = []

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(f_hook)
    try:
        # PyTorch 버전에 따라 full_backward_hook 사용
        try:
            h2 = target_layer.register_full_backward_hook(b_hook)
        except Exception:
            h2 = target_layer.register_backward_hook(b_hook)

        for p in img_paths:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            activations.clear()
            gradients.clear()

            # no_grad 끄고 실행
            with torch.enable_grad():
                out = model(x)
                logits = _pick_logits_from_output(out)  # [B]
                score = logits.view(-1)[0]

                model.zero_grad(set_to_none=True)
                score.backward(retain_graph=False)

                # Grad-CAM
                act = activations[0]          # [B,C,H,W]
                grad = gradients[0]           # [B,C,H,W]
                # detach 불필요 (enable_grad 블록 끝나면 자동 정리)
                weights = grad.mean(dim=(2, 3), keepdim=True)      # [B,C,1,1]
                cam = (weights * act).sum(dim=1, keepdim=True)     # [B,1,H,W]
                cam = torch.relu(cam)[0, 0].detach().cpu().numpy() # [H,W]

            # normalize & resize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, img.size)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            overlay = np.uint8(0.5 * heatmap + 0.5 * img_cv)

            out_path = os.path.join(out_dir, os.path.basename(p))
            _save_image(out_path, overlay)

    finally:
        h1.remove()
        try:
            h2.remove()
        except Exception:
            pass

    print(f"[xai] CAMs saved to {out_dir}")